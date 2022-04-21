import multiprocessing
import os
import random
import time
import urllib.parse
from collections import defaultdict
import gc
import pickle
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as PathosPool
from sentence_transformers import SentenceTransformer
import pandas as pd
from sqlitedict import SqliteDict
from hashids import Hashids
from scipy.spatial.distance import euclidean, cityblock
from fastdist.fastdist import euclidean as fdeuclidean
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import ClusterEnsembles as CE
import pymongo
from pymongo.errors import BulkWriteError, AutoReconnect
import re
from pprint import pprint

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import logging
from logging.handlers import RotatingFileHandler
log_name = 'vectorise_medium.py.log'
logging.basicConfig(filename=log_name, format='%(levelname)s:%(message)s', level=logging.DEBUG)
log = logging.getLogger()
handler = RotatingFileHandler(log_name, maxBytes=1048576)
log.addHandler(handler)

class VariableHandler(object):

    def __init__(self, model_path, id2vector_path, id2info_path, clustered_ids_path, mkmeans_model_path, num_items, hasher):
        self.model = self.set_model(model_path)
        self.id2vector_path = id2vector_path
        self.id2info_path = id2info_path
        self.clustered_ids_path = clustered_ids_path
        self.mkmeans_model_path = mkmeans_model_path
        self.num_items = num_items
        self.hasher = hasher

    def set_model(self, model_path):
        return SentenceTransformer(model_path)

    def distance(self, m1, m2):
        # return np.linalg.norm(m1-m2)
        return euclidean(m1,m2)

class VectorIndex():

    def __init__(self, vh):
        self.vh = vh        #VariableHandler object

    def remove_unsupported_punct(self, sentence):
        return sentence.replace('-', ' ').replace('!', '').replace('[', ' ').replace(']', ' ').replace(':', ' ')\
                .replace(';', ' ').replace('(', ' ').replace(')', ' ').replace('_', ' ').replace('   ', ' ')\
                .replace('  ', ' ').replace('    ', ' ').replace('. ', '.')

    def stringify_text(self, path):
        data = pd.read_csv(path)
        start = time.time()
        for i in range(len(data)):
            data._set_value(i, 'text', str(data.at[i, 'text']))
            if i % 10000 == 0:
                log.info('text string wrapped ... running time: {} seconds, {}/{}'.format(time.time() - start, i, len(data)))

        data.to_csv('{}_wrapped.csv'.format(path[:-4]))

    def generate_ids(self, size, batch_count):
        ids = []
        for num in range(size * batch_count, size * (batch_count + 1)):
            ids.append(self.vh.hasher.encode(num))

        return ids

    def are_same_vectors(self, v1, v2):
        return self.vh.distance(v1,v2) < 1e-4

    def search_query_fast(self, query, n_best):
        log.info(
            '<============================================= calling function search_query_fast =============================================>')
        search_start = time.time()
        start = time.time()
        id2vector = SqliteDict(self.vh.id2vector_path)
        id2info = SqliteDict(self.vh.id2info_path)
        lv1_clusters = SqliteDict(self.vh.clustered_ids_path)
        lv2_clusters = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/double_clusters_medium.sqlite')
        log.info('data loded in {} seconds!'.format(time.time() - start))

        log.info('encoding query!')
        start = time.time()
        encoded_query = self.vh.model.encode(query)
        log.info('query encoded in {} seconds!'.format(time.time() - start))

        ## find lv2 cluster closest to query
        start = time.time()
        centroids = [x['centroid'] for x in list(lv2_clusters.values())]
        d_to_lv2_clusters = [euclidean(encoded_query, c) for c in centroids]
        closest_lv2_cluster = sorted(dict(zip(d_to_lv2_clusters, list(lv2_clusters.keys()))).items())[0][1]
        log.info('found closest lv2 cluster %s in %.4f seconds!' % (closest_lv2_cluster, time.time() - start))

        ## find lv1 cluster closest to query using the closest lv2 cluster found above
        start = time.time()
        lv1_candidates = lv2_clusters[closest_lv2_cluster]['members']
        centroids = [lv1_clusters[candidate]['centroid'] for candidate in lv1_candidates]
        d_to_lv1_clusters = [euclidean(encoded_query, c) for c in centroids]
        closest_lv1_cluster = sorted(dict(zip(d_to_lv1_clusters, lv1_candidates)).items())[0][1]
        log.info('found closest lv1 cluster %s in %.4f seconds!' % (closest_lv1_cluster, time.time() - start))

        start = time.time()
        ids = lv1_clusters[closest_lv1_cluster]['members']
        log.info('retrieved ids in %.4f seconds!' % (time.time() - start))

        start = time.time()
        distances = [cityblock(encoded_query, id2vector[id]) for id in ids]
        log.info(distances)
        log.info('distance calculation finished in %.4f seconds!' % (time.time() - start))

        start = time.time()
        dist2id = sorted([[d, id] for d, id in zip(distances, ids)])[:n_best]
        log.info('sorted distance and id pairs in %.4f seconds!' % (time.time() - start))

        start = time.time()
        results = [[d,
                    id2info[id]['video_id'].decode('utf-8'),
                    '%02d:%02d' % (int(float(id2info[id]['timestamp']) // 60),
                                   int(float(id2info[id]['timestamp']) % 60)),
                    id2info[id]['text']] for d, id in dist2id
                   ]
        log.info('retrieved %d results in %.4f seconds!' % (len(results), time.time() - start))

        log.info('search finished in %.4f seconds!' % (time.time() - search_start))
        id2vector.close()
        id2info.close()
        lv1_clusters.close()
        lv2_clusters.close()

        return results

class DatabaseHandler():

    def __init__(self, vh):
        self.vh = vh    #VariableHandler object
        self.path_to_medium = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id2vector_medium.sqlite'
        self.db = self.connect_to_firestore()
        # self.db = self.connect_to_mongo()

    # find a point in data that is closest to a given point
    # return the point in vector form
    def find_pseudo_centroid(self, data, point):
        d = euclidean(data[0], point)  #
        nearest_point = None  # init nearest position

        for i in range(3, len(data)):
            temp_dist = euclidean(data[i], point)
            if temp_dist < d:
                log.info('found closer point! distance {}'.format(round(d,6)))
                d = temp_dist
                nearest_point = data[i]

        return nearest_point

    def incremental_fit_kmeans_model(self, n_clusters):
        log.info('<=================================== running minibatch kmeans ===================================>')
        id2vector = SqliteDict(self.vh.id2vector_path, autocommit=True)

        num_items = self.vh.num_items
        chunk_size = 300000

        start = time.time()
        log.info('generating script ids!')
        total_ids = list(map(self.vh.hasher.encode, list(range(num_items))))  #generate script ids by encoding integers
                                                                                #only using 23m, not full 41m
        log.info('script ids generated in %.4f seconds!' % (time.time()-start))

        mkmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                  random_state=4526,
                                  batch_size=16)

        for i in range((num_items // chunk_size) + 1):  # get indices to encode and retrieve vector from
            loop_start = time.time()
            ids_chunk = total_ids[i * chunk_size: (i + 1) * chunk_size]
            log.info('working on ids {}:{}!'.format(i * chunk_size, (i + 1) * chunk_size))

            start = time.time()
            vectors = list(map(id2vector.get,ids_chunk))
            log.info('mapped and converted vectors in %.4f seconds!'%(time.time()-start))
            log.info('starting to partial fit minibatch kmeans!')

            # partial fit kmeans on batch data
            start = time.time()
            mkmeans.partial_fit(vectors)
            log.info(
                'batched kmeans partial fit on batch data no. %d finished in %.4f seconds!'%(i, time.time() - start))

            log.info('loop duration: %.4f seconds'% (time.time() - loop_start))
            del vectors
            gc.collect()

        #### save mkmeans model to pickle
        with open(self.vh.mkmeans_model_path, 'wb') as f:
            pickle.dump(mkmeans, f)

        id2vector.close()

    def save_mkmeans_data(self):
        id2vector = SqliteDict(self.vh.id2vector_path)
        clustered_ids = SqliteDict(self.vh.clustered_ids_path, autocommit=True)

        with open(self.vh.mkmeans_model_path, 'rb') as f:
            mkmeans = pickle.load(f)

        num_items = self.vh.num_items
        chunk_size = 500000

        start = time.time()
        log.info('generating script ids!')
        total_ids = list(map(self.vh.hasher.encode, list(range(num_items))))  # generate script ids by encoding integers
        # only using 23m, not full 41m
        log.info('script ids generated in %.4f seconds!' % (time.time() - start))

        # get cluster centres
        centers = mkmeans.cluster_centers_

        # temporary dict to save database for this batch -> will only be i/o-ed once
        temp = defaultdict(list)

        for i in range((num_items // chunk_size) + 1):  # get indices to encode and retrieve vector from
            loop_start = time.time()
            ids_chunk = total_ids[i * chunk_size: (i + 1) * chunk_size]
            log.info('working on ids {}:{}!'.format(i * chunk_size, (i + 1) * chunk_size))

            start = time.time()
            vectors = list(map(id2vector.get,ids_chunk))
            log.info('mapped and retrieved vectors to a list in {} seconds!'.format(time.time()-start))
            log.info('starting to predict mkmeans labels for data batch!')

            # find labels of the batch by mkmeans
            labels = mkmeans.predict(vectors)
            log.debug(labels)
            log.info('saving data batch to a temporary dict!')
            # save data to a temporary dict
            for lb, iid in zip(labels, ids_chunk):
                temp[str(lb)].append(iid)
            log.info('batched saved! loop duration: {} seconds'.format(time.time() - loop_start))

            del vectors, ids_chunk
            gc.collect()

        # write data to sqlite
        log.info('writing data batches to the disk!')
        start = time.time()
        for key in temp.keys():
            clustered_ids['c'+key] = {'centroid':centers[int(key)],
                                      'members':temp[key]
                                      }

        log.info('wrote to database in {} seconds'.format(time.time() - start))

        id2vector.close()
        clustered_ids.close()

    def do_kmeans_double(self):
        double_clusters_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/double_clusters_medium.sqlite'
        double_clusters = SqliteDict(double_clusters_path, autocommit=True)

        clustered_ids = SqliteDict(self.vh.clustered_ids_path)

        cids = list(clustered_ids.keys())   # cluster ids from level-1 clusters
        centroids = [clustered_ids[key]['centroid'] for key in cids]    # centroids in level-1 clusters
        kms = KMeans(n_clusters=100, random_state=4526).fit(centroids)  # kmeans model to do level-2 clustering
        centers = kms.cluster_centers_                                  # level-2 clustering centroids

        higher_clusters = [lb for lb in kms.labels_]          # level-2 clustering labels for each level-1 cluster

        start = time.time()
        temp = defaultdict(list)
        for lb, cid in zip(higher_clusters, cids):          # temp[label] = [] add level-1 cluster ids (members) to level-2 clustering temp dict
            temp[lb].append(cid)
        log.info('finished clustering and allocating centroids to higher clusters in %.4f seconds!' % (time.time()-start))

        start = time.time()
        log.info('start writing to cc database!')
        for key in temp.keys():                             # write temp results to database in a better format
            double_clusters['cc'+str(key)] = {'centroid':centers[key],
                                                'members':temp[key]
                                            }
        log.info('wrote to database in %.4f seconds' % (time.time() - start))


    def test_mkmeans(self):
        clustered_ids = SqliteDict(self.vh.clustered_ids_path, autocommit=True)
        id2info = SqliteDict(self.vh.id2info_path, autocommit=True)

        for key in clustered_ids.iterkeys():
            print(key)
        # log.info(clustered_ids['c537']['members'])
        # log.info(clustered_ids['c37']['members'])

        clustered_ids.close()
        id2info.close()

    def test_double_kmeans(self):
        double_clusters_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/double_clusters_medium.sqlite'
        double_clusters = SqliteDict(double_clusters_path)
        id2info = SqliteDict(self.vh.id2info_path)

        for key in double_clusters.iterkeys():
            print(double_clusters[key])
            break
        # log.info(clustered_ids['c537']['members'])
        # log.info(clustered_ids['c37']['members'])

        double_clusters.close()
        id2info.close()

    def sort_mkmeans_data(self):
        clustered_ids = SqliteDict(self.vh.clustered_ids_path, autocommit=True)
        id2vector = SqliteDict(self.vh.id2vector_path, autocommit=True)
        run_start = time.time()

        for cluster in clustered_ids.iterkeys():
            log.info('sorting cluster id: {}, running time: {} seconds'.format(cluster, time.time()-run_start))
            cls = clustered_ids[cluster]
            centroid = cls['centroid']
            members = cls['members']
            cls['members'] = {}         #remove the list of members for update

            log.info('calculating distance from member vectors to centroid!')
            start = time.time()
            for id in members:
                v = id2vector[id]
                distance_to_centroid = self.vh.distance(centroid, v)    #find distance to centroid from each member vector
                cls['members'][id] = distance_to_centroid
            log.info('distance calculation finished in {} seconds!'.format(time.time()-start))

            # sort member vectors by distance to centroid
            members_sorted = {k: v for k, v in sorted(cls['members'].items(), key=lambda item: item[1])}
            cls['members'] = members_sorted

            #write to database
            clustered_ids[cluster] = cls
            log.info('vectors sorted by distance!')

    def convert_to_doc_for_mongo(self, scriptid, info_result, vector_result):
        return {
            'script_id': str(scriptid),
            'video_id': info_result['video_id'].decode('UTF-8'),
            'timestamp': info_result['timestamp'].decode('UTF-8'),
            'text': info_result['text'],
            'vector': vector_result.tolist()
        }

    def connect_to_mongo(self):
        # connet to MongdoDB serverless
        mongo_uri = 'mongodb+srv://{}:{}@serverlessttds3.lc7gy.mongodb.net/{}?retryWrites=true&w=majority'.format(
            urllib.parse.quote_plus('chanson76'),
            urllib.parse.quote_plus('Ahdrh212@@'),
            urllib.parse.quote_plus('ttds3_data')
        )
        return pymongo.MongoClient(mongo_uri)

    @staticmethod
    def write_to_mongo_parallel(batch):
        # connet to MongdoDB
        # mongo_uri = 'mongodb+srv://{}:{}@ttds3sample.lc7gy.mongodb.net/{}?retryWrites=true&w=majority'.format(
        #     urllib.parse.quote_plus('chanson76'),
        #     urllib.parse.quote_plus('Ahdrh212@@'),
        #     urllib.parse.quote_plus('test')
        # )
        mongo_uri = 'mongodb+srv://{}:{}@serverlessttds3.lc7gy.mongodb.net/{}?retryWrites=true&w=majority'.format(
            urllib.parse.quote_plus('chanson76'),
            urllib.parse.quote_plus('Ahdrh212@@'),
            urllib.parse.quote_plus('ttds3_data')
        )
        mongo_client = pymongo.MongoClient(mongo_uri)
        db = mongo_client.business
        db['id2data'].insert_many(batch)

    def add_to_mongo(self, resume=0, batch_size=10000, record_point=30000):
        id2vector = SqliteDict(self.vh.id2vector_path)
        id2info = SqliteDict(self.vh.id2info_path)
        log.info('[individual] loaded sqlite dictionaries!')

        total_rows = len(id2vector)

        # generate ids by encoding numbers
        start = time.time()
        ids = map(self.vh.hasher.encode, list(range(resume,total_rows)))
        log.info('took {} seconds to encode and append {} ids'.format(round(time.time() - start, 8), total_rows-resume))

        #init multiprocessing module
        # num_cores = multiprocessing.cpu_count()
        # p = PathosPool(num_cores)

        # empty list to append n Rows, which then will be appended to Avro file
        batch = []
        log.info('[individual] starting DB size {}!'.format(resume))
        log.info('[individual] start adding data to batch!')

        loop_start = time.time()

        #collect ids first to access values
        for i, id in enumerate(ids):
            batch.append(self.convert_to_doc_for_mongo(id, id2info[id], id2vector[id])) #retrieve info and append to data batch

            if i % record_point == 0:
                log.info('[individual] processing data ... {}/{} running time: {} seconds'.format(i+resume,
                                                                                                  total_rows,
                                                                                                  round(time.time() - loop_start, 4)))

            if i > 0 and i % batch_size == 0 or i+resume == total_rows:  # every N rows or at the end
                log.info('[individual] appended data to batches ... {}/{} running time: {} seconds'.format(
                    i+resume, total_rows, round(time.time() - loop_start, 4)
                ))
                write_start = time.time()
                try:
                    # connect to mongoDB, register IP address to Mongo before attempting to connect!!
                    client = self.connect_to_mongo()
                    db = client.business
                    log.info('[individual] successfully connected to MongoDB!')
                    db['id2data'].insert_many(batch)
                    client.close()
                    log.info('[individual] successfully saved to MongoDB! connection closed.')

                except AutoReconnect:
                    time.sleep(5)

                except:   #E11000 duplicate id error
                    client = self.connect_to_mongo()
                    db = client.business        #reconnect
                    log.info('MongoDB duplicate id error: removing duplicate ids and reconnecting!')
                    for doc in batch:
                        if '_id' in doc:
                            del doc['_id']
                    db['id2data'].insert_many(batch)
                    client.close()
                    log.info('[individual] successfully saved to MongoDB! connection closed.')

                log.info('wrote to MongoDB in {} seconds!'.format(time.time()-write_start))

                batch = []  # reset data_batch after appending N rows to file
                gc.collect()

        id2vector.close(force=True)
        id2info.close(force=True)

    def add_omitted_to_mongo(self, target_ids, resume=0, batch_size=10000, record_point=30000):
        id2vector = SqliteDict(self.vh.id2vector_path)
        id2info = SqliteDict(self.vh.id2info_path)
        log.info('[individual] loaded sqlite dictionaries!')

        total_rows = len(target_ids)

        # generate ids by encoding numbers
        start = time.time()
        ids = map(self.vh.hasher.encode, target_ids)
        log.info('took {} seconds to encode and append {} ids'.format(round(time.time() - start, 8), total_rows-resume))

        # empty list to append n Rows, which then will be appended to Avro file
        batch = []
        log.info('[individual] start adding data to batch!')

        loop_start = time.time()

        #collect ids first to access values
        for i, id in enumerate(ids):
            batch.append(self.convert_to_doc_for_mongo(id, id2info[id], id2vector[id])) #retrieve info and append to data batch

            if i % record_point == 0:
                log.info('[individual] processing data ... {}/{} running time: {} seconds'.format(i+resume,
                                                                                                  total_rows,
                                                                                                  round(time.time() - loop_start, 4)))

            if i > 0 and i % batch_size == 0 or i+resume == total_rows:  # every N rows or at the end
                log.info('[individual] appended data to batches ... {}/{} running time: {} seconds'.format(
                    i+resume, total_rows, round(time.time() - loop_start, 4)
                ))
                write_start = time.time()
                try:
                    # connect to mongoDB, register IP address to Mongo before attempting to connect!!
                    client = self.connect_to_mongo()
                    db = client.business
                    log.info('[individual] successfully connected to MongoDB!')
                    db['id2data'].insert_many(batch)
                    client.close()
                    log.info('[individual] successfully saved to MongoDB! connection closed.')

                except AutoReconnect:
                    time.sleep(5)

                except:   #E11000 duplicate id error
                    client = self.connect_to_mongo()
                    db = client.business        #reconnect
                    log.info('MongoDB duplicate id error: removing duplicate ids and reconnecting!')
                    for doc in batch:
                        if '_id' in doc:
                            del doc['_id']
                    db['id2data'].insert_many(batch)
                    client.close()
                    log.info('[individual] successfully saved to MongoDB! connection closed.')

                log.info('wrote to MongoDB in {} seconds!'.format(time.time()-write_start))

                batch = []  # reset data_batch after appending N rows to file
                gc.collect()

        id2vector.close(force=True)
        id2info.close(force=True)

    def format_cluster_ids_for_mongo(self, cid, details):
        return {'cluster_id':cid,
                'centroid':details['centroid'].tolist(),
                'members':details['members']}

    def add_cluster_ids_to_mongo(self):
        cluster_infos = SqliteDict(self.vh.clustered_ids_path)
        log.info('[individual] loaded sqlite dictionary!')

        # empty list to append n Rows, which then will be appended to Avro file
        batch = []
        log.info('[individual] start adding data to batch!')

        start = time.time()
        #collect ids first to access values
        ctr = 0
        for cluster in cluster_infos.iterkeys():
            batch.append(self.format_cluster_ids_for_mongo(cluster, cluster_infos[cluster])) #retrieve info and append to data batch

            log.info('[individual] processing data ... {}/{} running time: {} seconds'.format(ctr,
                                                                                                len(cluster_infos),
                                                                                                round(time.time() - start, 4)))
            ctr += 1

        write_start = time.time()
        try:
            # connect to mongoDB, register IP address to Mongo before attempting to connect!!
            client = self.connect_to_mongo()
            db = client.business
            log.info('[individual] successfully connected to MongoDB!')
            db['cluster_info'].insert_many(batch)
            client.close()
            log.info('[individual] successfully saved to MongoDB! connection closed.')

        except AutoReconnect:
            time.sleep(5)

        except:   #E11000 duplicate id error
            client = self.connect_to_mongo()
            db = client.business        #reconnect
            log.info('MongoDB duplicate id error: removing duplicate ids and reconnecting!')
            for doc in batch:
                if '_id' in doc:
                    del doc['_id']
            db['cluster_info'].insert_many(batch)
            client.close()
            log.info('[individual] successfully saved to MongoDB! connection closed.')

        log.info('wrote to MongoDB in {} seconds!'.format(time.time()-write_start))

        cluster_infos.close()

    def delete_from_mongo(self):
        # connect to mongoDB, register IP address to Mongo before attempting to connect!!
        db = self.connect_to_mongo()
        db['id2data'].drop()
        db['id2vector'].drop()
        # x = id2data_col.delete_many({})
        # log.info('[individual] deleted {} rows!'.format(x.deleted_count))

    @staticmethod
    def get_script_id(search_result):
        return search_result['script_id']

    def test_mongo(self):
        total_length = 40356731

        #regex for searching everything
        regx = re.compile('^.*', re.IGNORECASE)
        # test_id = self.vh.hasher.encode(194)
        db = self.connect_to_mongo().business
        log.info('connected to MongoDB!')

        log.info('start searching!')
        search_start = time.time()
        items = db['id2data'].find({'script_id': regx})
        log.info('searched finished in {} seconds!'.format(time.time()-search_start))

        # init multiprocessing module
        num_cores = multiprocessing.cpu_count()

        with Pool(num_cores, maxtasksperchild=1000) as p:

            retrieve_start = time.time()
            script_ids = p.imap_unordered(self.get_script_id, items)
            log.info('parallel id retrieval took {} seconds!'.format(round(time.time()-retrieve_start,4)))

            log.info('converting script ids map object to set!')
            start = time.time()
            script_ids_set = {*script_ids}
            log.info('map object converted to set in {} seconds!'.format(round(time.time()-start,4)))

            full_id_start = time.time()
            all_ids_encoded = {*map(self.vh.hasher.encode, list(range(total_length)))}    #this will take roughly 13 minutes
            log.info('took {} seconds to create a full ids set!'.format(time.time() - full_id_start))

            xor_start = time.time()
            ids_omitted = all_ids_encoded ^ script_ids_set
            log.info('ommited ids retrieved in {} seconds!'.format(time.time()-xor_start))

            with open('ids_omitted.pkl', 'wb') as f:
                pickle.dump(sorted(ids_omitted), f)

    def test_search_mongo(self, query, distance_threshold=3.4, n_best=10):

        log.info('encoding query!')
        start = time.time()
        encoded_query = self.vh.model.encode(query)
        log.info('query encoded in {} seconds!'.format(time.time() - start))

        # find closest cluster by calculating distance to centroid
        start = time.time()
        closest_d_cluster = 100
        closest_cluster = -1

        # regex for searching everything
        regx = re.compile('^.*', re.IGNORECASE)
        db = self.connect_to_mongo().business
        log.info('connected to MongoDB!')

        log.info('start searching!')
        search_start = time.time()
        clusters_cursor = db['cluster_info'].find({'cluster_id': regx}, {'_id':0, 'members':0})   #don't retrieve id and members for fast retreival
        log.info('cluster retreival finished in {} seconds!'.format(time.time() - search_start))

        cluster_start = time.time()
        for cursor in clusters_cursor:
            # find distance between encoded query and the centroid of each cluster
            distance = self.vh.distance(encoded_query, cursor['centroid'])

            if distance < closest_d_cluster:
                # log.info('updated closest point! distance: {} | running time: {} seconds'.format(distance, time.time() - start))
                closest_d_cluster = distance
                closest_cluster = cursor['cluster_id']

        log.info('found closest cluster in {} seconds!'.format(time.time() - cluster_start))
        print(closest_cluster)

        # result_texts = {}
        # members = db['cluster_info'].find({'cluster_id':closest_cluster}, {'_id':0, 'centroid':0})[0]['members']
        #
        # for script_id in members:
        #     #there should be a single unique item for each script id, hence [0]
        #     data = db['id2data'].find({'script_id':script_id}, {'_id':0, 'video_id':0, 'timestamp':0, 'text':0})[0]
        #
        #     distance = self.vh.distance(encoded_query, data['vector'])
        #     if distance < distance_threshold:
        #         result_texts[script_id] = distance
        #         # only get the best n results
        #         if len(result_texts) >= n_best:
        #             break

        #  # log.info('updated closest text! distance: {} | running time: {} seconds'.format(distance, time.time() - start))
        # result_texts[str(distance)] = {'video_id': data['video_id'],
        #                                 'timestamp': '{}:{}'.format(int(float(data['timestamp']) // 60),
        #                                                             int(float(data['timestamp']) % 60)),
        #                                 'text': data['text']}
        #
        # result_texts = sorted(result_texts.items())
        log.info('search finished in {} seconds!'.format(round(time.time() - search_start, 4)))
        # print(result_texts)
        #
        # for i, result in enumerate(result_texts):
        #     log.info(result)


    def connect_to_firestore(self):
        # export key.json for auth
        os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/chanwoo/PycharmProjects/inf4-ttds-cw3/' \
                                                     'ttds3-339414-firebase-adminsdk-qhmye-ac751f18fd.json'

        project_id = 'ttds3-339414'
        # Use the application default credentials
        cred = credentials.ApplicationDefault()
        try:
            firebase_admin.initialize_app(cred, {
                'projectId': project_id,
            })
        except ValueError:
            firebase_admin.get_app(name='[DEFAULT]')

        return firestore.client()

    def initialise_firestore(self):
        id2vector = SqliteDict(self.vh.id2vector_path)
        id2info = SqliteDict(self.vh.id2info_path)
        start_id = self.vh.hasher.encode(0)

        db = self.connect_to_firestore()
        doc_ref = db.collection('id2data').document(start_id)
        doc_ref.set(self.convert_to_doc_for_firestore(
                                                        script_id=start_id,
                                                        info_result=id2info[start_id],
                                                        vector_result=id2vector[start_id]
                                                      )
                    )

        id2info.close(force=True)
        id2vector.close(force=True)

    def convert_to_doc_for_firestore(self, script_id, info_result, vector_result):
        return {
                'script_id': script_id,
                'video_id': info_result['video_id'].decode('UTF-8'),
                'timestamp': info_result['timestamp'].decode('UTF-8'),
                'text': info_result['text'],
                'vector': vector_result.tolist()
            }

    @staticmethod
    # made this function static because of the multiprocessing doesn't understand non-static pickling
    def fs_parallel_upload_id2data(doc):
        fire_db = firestore.client()
        fire_db.collection('id2data').document(doc['script_id']).set(doc)

    @staticmethod
    # made this function static because of the multiprocessing doesn't understand non-static pickling
    def fs_parallel_delete_id2data(doc_id):
        fire_db = firestore.client()
        try:
            fire_db.collection('id2data').document(doc_id).delete()
        except:
            log.info('document %s is already gone' % (doc_id))
            pass

    def delete_from_firestore(self, resume=0):
        id2vector = SqliteDict(self.vh.id2vector_path)

        # generate ids by encoding numbers
        total_rows = len(id2vector)

        # init multiprocessing module
        num_cores = multiprocessing.cpu_count()
        p = PathosPool(num_cores)

        delete_start = time.time()

        batch_size = 1000000
        for x in range(((total_rows-resume)//batch_size) + 1):
            log.info('[individual] generating ids!')
            start = time.time()
            ids = list(map(self.vh.hasher.encode, list(range(resume+ x * batch_size, resume + (x+1) * batch_size))))
            log.info('[individual] took %.4f seconds to encode and append ids [%d:%d]' % (time.time() - start, resume+ x * batch_size,
                                                                                          resume + (x+1) * batch_size))

            start = time.time()
            p.map(self.fs_parallel_delete_id2data, ids)
            log.info('deleted %d items in %.4f seconds!' % (time.time()-start, batch_size))
            log.info('running time: %.4f | current time: %s' % (time.time()-delete_start, time.ctime(time.time())))

    # @staticmethod
    def add_to_firestore(self, batch_size, resume=0, stop=False):

        id2vector = SqliteDict(self.vh.id2vector_path)
        id2info = SqliteDict(self.vh.id2info_path)
        log.info('[individual] loaded sqlite dictionaries!')

        # generate ids by encoding numbers
        start = time.time()
        total_rows = len(id2vector)
        ids = map(self.vh.hasher.encode, list(range(resume,total_rows)))
        log.info('[individual] took %.4f seconds to encode and append %d ids' % (time.time() - start, total_rows-resume))

        # init multiprocessing module
        num_cores = multiprocessing.cpu_count()
        p = PathosPool(num_cores)

        # empty list to append n Rows, which then will be appended to Avro file
        data_batch = []
        log.info('[individual] start moving data to firestore at %s!' % (time.ctime(time.time())))

        start = time.time()

        # collect ids first to access values
        for i, id in enumerate(ids):
            # append to data batch which are to be added to file every N
            data_batch.append(self.convert_to_doc_for_firestore(id, id2info[id], id2vector[id]))
            # now append to mongoDB
            if i > 0 and i % batch_size == 0 or resume + i == total_rows:  # every N rows or at the end
                log.info('[individual] appended data to temporary batch, currently at: %d/%d running time: %.4f seconds' % (
                    i+resume, total_rows, time.time() - start
                ))
                p.map(self.fs_parallel_upload_id2data, data_batch)
                log.info('[individual] performed parallelised writes to Firestore, currently at %d rows ... running time: %.4f seconds\ntime:%s' % (
                    resume + i,
                    time.time() - start,
                    time.ctime(time.time())))
                data_batch = []  # reset data_batch after appending N rows to file
                gc.collect()

                #use when doing only 1 loop
                if stop:
                    break

        id2vector.close(force=True)
        id2info.close(force=True)

    def split_cluster_data_to_batches_fs(self, doc):
        num_members = len(doc['members'])
        log.info('num members in cluster {}: {}'.format(doc['cluster_id'], num_members))
        batches = []
        max_size = 2**14
        if num_members > max_size:
            for x in range((num_members//max_size)+1):
                batches.append(
                    {'cluster_id': doc['cluster_id'],
                     'members': doc['members'][max_size * x : max_size * (x+1)]
                     }
                )
        else:
            batches.append(doc)

        return batches

    # the medium one has less than 3000 items so just upload to firestore without batching
    def add_clusters_to_firestore(self):
        lv1_clusters = SqliteDict(self.vh.clustered_ids_path)
        lv2_clusters_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/double_clusters_medium.sqlite'
        lv2_clusters = SqliteDict(lv2_clusters_path)
        log.info('[individual] loaded sqlite dictionaries!')

        # connect to firestore
        db = self.connect_to_firestore()
        log.info('[individual] connected to firestore, start adding data to firestore!')

        start = time.time()
        #collect ids first to access values
        ctr = 0
        for cluster in lv1_clusters.iterkeys():

            doc = lv1_clusters[cluster]
            doc['centroid'] = doc['centroid'].tolist()
            log.info(len(doc['members']))
            db.collection('lv1_clusters').document(cluster).set(doc)

            log.info('[individual] uploaded lv1 cluster to firestore! ... %d/%d running time: %.4f seconds' % (
                                                                                                ctr,
                                                                                                len(lv1_clusters),
                                                                                                time.time() - start))
            ctr += 1

        log.info('[individual] uploaded lv1 cluster to firestore in %.4f seconds!' % (time.time()-start) )
        #upload lv2 documents: should be brief as there are only 100 clusters
        start = time.time()
        for cluster in lv2_clusters.iterkeys():
            #upload lv2 cluster doc to collection
            doc = lv2_clusters[cluster]
            doc['centroid'] = doc['centroid'].tolist()
            log.info(len(doc['members']))
            db.collection('lv2_clusters').document(cluster).set(doc)
        # log.info('uploaded lv2 clusters in %.4f seconds!' % (time.time() - start))

        log.info('uplaoded all centroids to firestore in {} seconds!'.format(time.time() - start))
        log.info('------------------------------------------------------------------------------')
        lv1_clusters.close()
        lv2_clusters.close()

    def add_centroids_to_firestore(self):
        cluster_infos = SqliteDict(self.vh.clustered_ids_path)
        log.info('[individual] loaded sqlite dictionary!')

        # connect to firestore
        db = self.connect_to_firestore()
        log.info('[individual] start adding data to firestore!')

        start = time.time()
        #collect ids first to access values
        ctr = 0
        fs_batch = db.batch()
        for cluster in cluster_infos.iterkeys():
            ctr += 1

            log.info('[individual] processing document no.{}/{}'.format(ctr, len(cluster_infos)))
            doc = {'centroid':cluster_infos[cluster]['centroid'].tolist()}

            doc_ref = db.collection('clusters').document(cluster)
            fs_batch.set(doc_ref, doc)

            if ctr % 200 == 0:
                fs_batch.commit()
                fs_batch = db.batch()   #reset batch

            log.info('[individual] uploaded centroids to firestore! ... {}/{} running time: {} seconds'.format(
                                                                                                ctr,
                                                                                                len(cluster_infos),
                                                                                                round(time.time() - start, 4)))


        log.info('uplaoded all centroids to firestore in {} seconds!'.format(time.time() - start))
        log.info('------------------------------------------------------------------------------')
        cluster_infos.close()

    def add_cluster_members_to_firestore(self):
        cluster_infos = SqliteDict(self.vh.clustered_ids_path)
        log.info('[individual] loaded sqlite dictionary!')

        # connect to firestore
        db = self.connect_to_firestore()
        log.info('[individual] start adding data to firestore!')

        start = time.time()
        #collect ids first to access values
        ctr = 0
        for cluster in cluster_infos.iterkeys():
            ctr += 1

            log.info('[individual] processing document no.{}/{}'.format(ctr, len(cluster_infos)))
            doc = {'cluster_id':cluster,
                   'members':cluster_infos[cluster]['members']}

            #split document into multiple documents based on the number of members it contains (due ot firestore doc size limit 20k)
            doc_batches = self.split_cluster_data_to_batches_fs(doc)
            log.info('cluster member data splitted into batches!')

            if len(doc_batches) == 1:   # if there are less than threshold members (16384)
                db.collection('cluster_members').document(cluster + '_b1').set(doc_batches[0])
            else:
                fs_batch = db.batch()
                for i, d in enumerate(doc_batches):
                    doc_ref = db.collection('cluster_members').document(cluster+'_b'+str(i+1))
                    fs_batch.set(doc_ref, d)

                fs_batch.commit()
            log.info('[individual] converted item to be in firestore format and uploaded ... {}/{} running time: {} seconds'.format(ctr,
                                                                                                len(cluster_infos),
                                                                                                round(time.time() - start, 4)))


        log.info('all cluster members data uploaded to firestore in {} seconds!'.format(time.time() - start))
        log.info('------------------------------------------------------------------------------')
        cluster_infos.close()

    def test_firestore(self, test_run=0):
        # 20199
        test_id = self.vh.hasher.encode(test_run)
        db = self.connect_to_firestore()
        id2data = db.collection('id2data')  # get collection object
        doc = id2data.document(test_id)  # get document object from collection by key
        data = doc.get().to_dict()
        try:
            print(data['video_id'])
            print(data['timestamp'])
            print(data['text'])
        except TypeError:
            raise KeyError('key not found! this id has not been uploaded yet!')



# static functions for flask app
def flatten(nd_list):
    return [item for sublist in nd_list for item in sublist]

def ddd(x,y):
    return (x-y) ** 2

def search_query_fast(encoded_query, n_best=300, threshold=10):
    id2vector_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite'
    id2info_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_info.sqlite'
    lv1_clusters_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/clusters_medium.sqlite'
    lv2_clusters_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/double_clusters_medium.sqlite'

    start = time.time()
    with open('lv2_keys.pkl', 'rb') as f:
        lv2_keys = pickle.load(f)
    log.info('[individual] loaded lv2 keys in %.4f seconds!' % (time.time()-start))

    search_start = time.time()
    # load data
    start = time.time()
    id2vector = SqliteDict(id2vector_path)
    id2info = SqliteDict(id2info_path)
    lv1_clusters = SqliteDict(lv1_clusters_path)
    lv2_clusters = SqliteDict(lv2_clusters_path)
    log.info('[individual] data loaded in %.4f seconds!' % (time.time() - start))

    ## find lv2 cluster closest to query
    start = time.time()
    centroids = [lv2_clusters[x]['centroid'] for x in lv2_keys]
    log.info('[individual] %d centroids in lv2' % (len(centroids)))
    d_to_lv2_clusters = list(map(cityblock, [encoded_query]*len(centroids), centroids))
    closest_lv2_cluster = sorted(dict(zip(d_to_lv2_clusters, lv2_keys)).items())[0][1]
    log.info('[individual] found closest lv2 cluster %s in %.4f seconds!' % (closest_lv2_cluster, time.time() - start))

    ## find lv1 cluster closest to query using the closest lv2 cluster found above
    start = time.time()
    lv1_candidates = lv2_clusters[closest_lv2_cluster]['members']
    centroids = [lv1_clusters[candidate]['centroid'] for candidate in lv1_candidates]
    log.info('[individual] %d centroids in lv1' % (len(centroids)))
    d_to_lv1_clusters = list(map(cityblock, [encoded_query]*len(centroids), centroids))
    closest_lv1_cluster = sorted(dict(zip(d_to_lv1_clusters, lv1_candidates)).items())[0][1]
    log.info('[individual] found closest lv1 cluster %s in %.4f seconds!' % (closest_lv1_cluster, time.time() - start))

    start = time.time()
    ids = lv1_clusters[closest_lv1_cluster]['members']
    vs = [id2vector[id] for id in ids]
    log.info('[individual] retrieved ids in %.4f seconds!' % (time.time()-start))

    start = time.time()
    distances = list(map(fdeuclidean, [encoded_query]*len(ids), vs))
    # distances = list(map(fdeuclidean, [encoded_query]*len(ids), id2[fdeuclidean(encoded_query, id2vector[id]) for id in ids]
    log.info('[individual] fd distance calculation finished in %.4f seconds!' % (time.time()-start))

    start = time.time()
    dist2id = sorted([[d,id] for d, id in zip(distances, ids)])[:n_best]
    log.info('[individual] sorted distance and id pairs in %.4f seconds!' % (time.time()-start))

    start = time.time()
    results = [[float(d),
                id2info[id]['video_id'].decode('utf-8'),
                '%02d:%02d' % (int(float(id2info[id]['timestamp']) // 60),
                               int(float(id2info[id]['timestamp']) % 60)),
                id2info[id]['text']] for d, id in dist2id
               ]
    log.info('[individual] retrieved %d results in %.4f seconds!' % (len(results), time.time() - start))


    log.info('[individual] search finished in %.4f seconds!' % (time.time() - search_start))
    id2vector.close()
    id2info.close()
    lv1_clusters.close()
    lv2_clusters.close()
    return sorted(results)

    # ### find member vectors and sort them by distance
    # start = time.time()
    # result_dict = {}
    # for id in lv1_clusters[closest_lv1_cluster]['members']:
    #     distance = fdeuclidean(encoded_query, id2vector[id])
    #     if distance < threshold:
    #         # log.info('updated closest text! distance: {} | running time: {} seconds'.format(distance, time.time() - start))
    #         result_dict[str(distance) + str(time.time())[-4:]] = {
    #             'video_id': id2info[id]['video_id'].decode('utf-8'),
    #             'timestamp': '%d:%d' % (int(float(id2info[id]['timestamp']) // 60),
    #                                     int(float(id2info[id]['timestamp']) % 60)),
    #             'text': id2info[id]['text']}
    #         # only get the best n results
    #         if len(result_dict) >= n_best:
    #             break
    #
    # result_dict = sorted(result_dict.items())
    # log.info('[individual] results retrieved and sorted %d results in %.4f seconds!' % (len(result_dict), time.time() - start))
    # log.info('[individual] search finished in %.4f seconds!' % (time.time() - search_start))
    # id2vector.close()
    # id2info.close()
    # lv1_clusters.close()
    # lv2_clusters.close()
    #
    # return result_dict

def search_query_fast_firestore(encoded_query, n_best=300, threshold=65):

    search_start = time.time()

    # find closest cluster by calculating distance to centroid
    start = time.time()
    closest_d_cluster = 100
    closest_cluster = 'cNotFound'

    # regex for searching everything
    db = connect_to_firestore()
    log.info('connected to Firestore!')

    log.info('start searching!')
    search_start = time.time()
    lv2_clusters = db.collection('lv2_clusters').stream()
    log.info('cluster retreival finished in %.4f seconds!' % (time.time() - search_start))

    ## find lv2 cluster closest to query
    start = time.time()
    docs = [(doc.id, doc.to_dict()) for doc in lv2_clusters]
    d_to_lv2_clusters = [(cityblock(encoded_query, doc['centroid']), ccid) for ccid, doc in docs]
    closest_lv2_cluster = sorted(d_to_lv2_clusters)[0][1]
    log.info('found closest lv2 cluster %s in %.4f seconds!' % (closest_lv2_cluster, time.time() - start))

    ## find lv1 cluster closest to query using the closest lv2 cluster found above
    start = time.time()
    doc_ref = db.collection('lv2_clusters').document(closest_lv2_cluster)   #retrieve lv1 cluster members using the closest lv2 cluster
    lv1_candidates = doc_ref.get().to_dict()['members']                     # less than 100 lv1 cluster members, fast until here
    closest_d = 1000
    closest = 'c000'

    lv1_centroids = [(candid, db.collection('lv1_clusters').document(candid).get().to_dict()['centroid']) for candid in lv1_candidates]
    d_to_lv1_clusters = [(cityblock(encoded_query, centroid), id) for id, centroid in lv1_centroids]    #calculate cityblock distance from lv1 centroids to query
    closest_lv1_cluster = sorted(d_to_lv1_clusters)[0][1]
    log.info('found closest lv1 cluster %s in %.4f seconds!' % (closest_lv1_cluster, time.time() - start))
    # log.info('found closest lv1 cluster %s in %.4f seconds!' % ('casd', time.time() - start))

    # start = time.time()
    # ids = lv1_clusters[closest_lv1_cluster]['members']
    # log.info('retrieved ids in %.4f seconds!' % (time.time()-start))
    #
    # start = time.time()
    # distances = [cityblock(encoded_query, id2vector[id]) for id in ids]
    # log.info('distance calculation finished in %.4f seconds!' % (time.time()-start))
    #
    # start = time.time()
    # dist2id = sorted([[d,id] for d, id in zip(distances, ids)])[:n_best]
    # log.info('sorted distance and id pairs in %.4f seconds!' % (time.time()-start))
    #
    # start = time.time()
    # results = [[float(d),
    #             id2info[id]['video_id'].decode('utf-8'),
    #             '%02d:%02d' % (int(float(id2info[id]['timestamp']) // 60),
    #                            int(float(id2info[id]['timestamp']) % 60)),
    #             id2info[id]['text']] for d, id in dist2id
    #            ]
    # log.info('retrieved %d results in %.4f seconds!' % (len(results), time.time() - start))
    #
    #
    # log.info('search finished in %.4f seconds!' % (time.time() - search_start))
    # id2vector.close()
    # id2info.close()
    # lv1_clusters.close()
    # lv2_clusters.close()
    # return results

    # ### find member vectors and sort them by distance
    # start = time.time()
    # result_dict = {}
    # for id in lv1_clusters[closest_lv1_cluster]['members']:
    #     distance = cityblock(encoded_query, id2vector[id])
    #     if distance < threshold:
    #         # log.info('updated closest text! distance: {} | running time: {} seconds'.format(distance, time.time() - start))
    #         result_dict[str(distance) + str(time.time())[-4:]] = {
    #             'video_id': id2info[id]['video_id'].decode('utf-8'),
    #             'timestamp': '%d:%d' % (int(float(id2info[id]['timestamp']) // 60),
    #                                     int(float(id2info[id]['timestamp']) % 60)),
    #             'text': id2info[id]['text']}
    #         # only get the best n results
    #         if len(result_dict) >= n_best:
    #             break
    #
    # result_dict = sorted(result_dict.items())
    # log.info('results retrieved and sorted %d results in %.4f seconds!' % (len(result_dict), time.time() - start))
    # log.info('search finished in %.4f seconds!' % (time.time() - search_start))
    # id2vector.close()
    # id2info.close()
    # lv1_clusters.close()
    # lv2_clusters.close()
    #
    # return result_dict


def connect_to_firestore():
    # export key.json for auth
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/chanwoo/PycharmProjects/inf4-ttds-cw3/' \
                                                'ttds3-339414-firebase-adminsdk-qhmye-ac751f18fd.json'

    project_id = 'ttds3-339414'
    # Use the application default credentials
    cred = credentials.ApplicationDefault()
    try:
        firebase_admin.initialize_app(cred, {
            'projectId': project_id,
        })
    except ValueError:
        firebase_admin.get_app(name='[DEFAULT]')

    return firestore.client()

def search_query_firestore(query, distance_threshold, n_best):

    log.info('encoding query!')
    start = time.time()
    model = SentenceTransformer('snt_tsfm_model/')
    encoded_query = model.encode(query)
    log.info('query encoded in {} seconds!'.format(time.time() - start))

    # find closest cluster by calculating distance to centroid
    start = time.time()
    closest_d_cluster = 100
    closest_cluster = 'cNotFound'

    # regex for searching everything
    db = connect_to_firestore()
    log.info('connected to Firestore!')

    log.info('start searching!')
    search_start = time.time()
    centroids = db.collection('clusters').stream()
    log.info('cluster retreival finished in %.4f seconds!' % (time.time() - search_start))

    cluster_start = time.time()
    read_count = 0
    for c in centroids:
        read_count += 1
        doc = c.to_dict()
        # find distance between encoded query and the centroid of each cluster
        distance = euclidean(encoded_query, doc['centroid'])

        if distance < closest_d_cluster:
            log.info('updated closest point! distance: %.4f | running time: %.4f seconds'% (distance, time.time() - start))
            closest_d_cluster = distance
            closest_cluster = c.id

    log.info('found closest cluster in %.4f seconds!' % (time.time() - cluster_start))
    log.info('read count: %d' % (read_count))

    member_ids = []
    for i in range(1,20):
        cid = closest_cluster + '_b%s' % (str(i))
        try:
            doc_ref = db.collection('cluster_members').document(cid)
            doc = doc_ref.get().to_dict()
            member_ids.append(doc['members'])
        except:
            # log.info('end of cluster batches reached!')
            pass

    result_dict = {}
    member_ids = flatten(member_ids)

    for id in member_ids:
        doc_ref = db.collection('id2data').document(id)
        doc = doc_ref.get().to_dict()

        try: # currently there are 16m documents so there should be member ids that are not found
            distance = euclidean(encoded_query, doc['vector'])
            if distance < distance_threshold:
                result_dict[str(distance)] = {'video_id': doc['video_id'],
                                                'timestamp': '%02d:%02d' % (int(float(doc['timestamp']) // 60),
                                                                        int(float(doc['timestamp']) % 60)),
                                                'text': doc['text']}
                log.info('result found! current results: %d' % (len(result_dict)))
                # only get the n best results
                if len(result_dict) >= n_best:
                    break
        except:
            pass

    result_texts = sorted(result_dict.items())
    log.info('search finished in %.4f seconds!' % (round(time.time() - search_start, 4)))

    return result_texts


class Clustering():

    def __init__(self, vh):
        self.vh = vh

    def extract_sample_vectors(self, sample_num):
        id2vector = SqliteDict(self.vh.id2vector_path)
        # id2info = SqliteDict(self.vh.id2info_path)

        randints = random.sample(range(self.vh.num_items),sample_num)
        rand_ids = map(self.vh.hasher.encode, randints)

        vectors = []
        for id in rand_ids:
            vectors.append(id2vector[id])

        return vectors


def main():
    # model = SentenceTransformer('stsb-mpnet-base-v2')
    # model.save('snt_tsfm_model/')

    vh = VariableHandler(
        model_path='snt_tsfm_model/',
        id2vector_path='/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite',
        id2info_path='/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_info.sqlite',
        clustered_ids_path='/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/clusters_medium.sqlite',
        mkmeans_model_path='/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/mkmeans_model_medium.pkl',
        num_items=22630575,
        hasher=Hashids(salt='eerotke hameon jeoldaemorugetzi', min_length=7)
    )

    vi = VectorIndex(vh)
    # vi.test_data()

    ###### test created dataset
    # vi.vectorised_data_test(mode='vector', test_type='key')
    # vi.vectorised_data_test(mode='info', test_type='key')
    # vi.vectorised_data_test(test_type='vectorisation',size=10000)

    query = 'you need to try a completely new query'
    # vi.search_query_fast(query=query,
    #                       n_best=1000)
    # model = SentenceTransformer('snt_tsfm_model')
    # encoded_query = model.encode(query)
    # search_query_fast(encoded_query=encoded_query, n_best=1000, threshold=10)

    dh = DatabaseHandler(vh)
    # encoded_query = dh.vh.model.encode(query)
    # search_query_fast(encoded_query)
    # search_query_fast_firestore(encoded_query)

    """""""""incremental kmeans"""""""""
    # dh.incremental_fit_kmeans_model(10000)
    # dh.save_mkmeans_data()
    # dh.do_kmeans_double()
    # dh.test_double_kmeans()
    # dh.test_mkmeans()
    # dh.sort_mkmeans_data()


    """""""""Firestore"""""""""
    # dh.initialise_firestore()
    # dh.add_to_firestore(batch_size=50000, resume=22630575, stop=False)
    # dh.test_firestore(19700000)
    # dh.add_clusters_to_firestore()              # upload 2-level clusters
    # dh.add_centroids_to_firestore()           # for 1-level clustering structure there are too many members per cluster
    # dh.add_cluster_members_to_firestore()     # so they should be divided into batches
    # search_query_firestore(query=query, n_best=10, distance_threshold=4)
    dh.delete_from_firestore(16365710)

    """""""""mongoDB (depricated) """""""""
    # dh.add_to_mongo(resume=11578909, batch_size=350000)
    # omitted = []
    # dh.add_omitted_to_mongo(target_ids=omitted, batch_size=300000)
    # dh.delete_from_mongo()
    # dh.test_mongo()
    # dh.add_cluster_ids_to_mongo()
    # dh.test_search_mongo(query=query)

    # clu = Clustering(vh)

    #do the following to access google compute engine
    # gcloud compute ssh --zone "europe-west2-c" "ttdsvm"  --project "ttds3-339414"
if __name__ == '__main__':
    main()