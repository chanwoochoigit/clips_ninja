import gc
import pickle
import time
import sys
from random import randrange
import logging
import numpy as np
import multiprocessing
from fastdist.fastdist import euclidean as fdeuclidean
from fastdist.fastdist import sqeuclidean as fdsqeuclidean
from fastdist.fastdist import cityblock as fdcityblock
from fastdist.fastdist import cosine as fdcosine
from fastdist.fastdist import hamming as fdhamming

from pathos.multiprocessing import ProcessingPool as PathosPool
from scipy.spatial.distance import euclidean, braycurtis, canberra,\
                                   chebyshev, cityblock, correlation,\
                                   cosine, euclidean, jensenshannon,\
                                   minkowski,\
                                   sqeuclidean, wminkowski
from pyspark import SparkConf, SparkContext
from math import hypot
from sentence_transformers import SentenceTransformer
from itertools import product
from sqlitedict import SqliteDict
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from hashids import Hashids
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession
import pandas as pd
import random
from py_mini_racer import py_mini_racer
import json
from kmeans_gpu import KMeans as kmeans_gpu
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sent2vec

from logging.handlers import RotatingFileHandler

log_name = 'play_around.py.log'
logging.basicConfig(filename=log_name, format='%(levelname)s:%(message)s', level=logging.DEBUG)
log = logging.getLogger()
handler = RotatingFileHandler(log_name, maxBytes=1048576)
log.addHandler(handler)

hasher = Hashids(salt='eerotke hameon jeoldaemorugetzi', min_length=7)
stuff = ['abcde']*10000
compare = ['abcde']*5000+['cdfaf']*5000

def play_1():
    counter = 0
    start = time.time()
    for s, c in zip(stuff, compare):
        sys.stdout.write('\r')
        sys.stdout.write(
            'validating keys ... {}/{} running time: {} seconds'.format(counter, len(compare), round(time.time() - start, 4)))
        sys.stdout.flush()
        counter += 1
        if s != c:
            logging.warning('\nwrong!!')

def play_2():
    start = 0
    stop = 1000000
    size = 100
    randnums = [randrange(start,stop) for iter in range(size)]
    print(randnums)

def play_3():
    id2vector_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite'
    id2vector = SqliteDict(id2vector_path, autocommit=True)
    ctr = 0
    start = time.time()
    vectors = []
    ids = []
    for id, v in id2vector.iteritems():
        if ctr >= 5000000:
            break
        vectors.append(v)
        ids.append(id)
        if ctr % 100000 == 0 :
            logging.info('creating vector and id list! {}/{} | time spent: {} seconds'.format(ctr, 42000000, time.time() - start))
        ctr += 1

    batch_size = 16
    chunk_size = 1000
    pca = IncrementalPCA(n_components=3, batch_size=batch_size, copy=False)
    logging.info('starting incremental pca!')
    start = time.time()
    for x in range((len(vectors) // batch_size) + 1):
        # divide data into batches according to batch size:
        _from = batch_size * x
        _to = batch_size * (x + 1)
        if batch_size * (x + 1) > len(vectors):
            _to = len(vectors)
        batch = vectors[_from:_to]

        if x % batch_size*100 == 0:
            logging.info('starting to perform pca on data batch [{}:{}]'.format(_from, _from+(_to-_from)*100))
        if len(batch) != 0:
            pca.partial_fit(batch)
        if x % batch_size*100 == 0:
            logging.info('done partial pca! | time spent: {} seconds'.format(time.time() - start))

    # id2pca_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_pca.sqlite'
    # id2pca = SqliteDict(id2pca_path, autocommit=True)
    # start = time.time()
    # for x in range((len(vectors) // batch_size) + 1):
    #     # divide data into batches according to batch size:
    #     _from = batch_size * x
    #     _to = batch_size * (x + 1)
    #     if batch_size * (x + 1) > len(vectors):
    #         _to = len(vectors)
    #     batch = vectors[_from:_to]
    #     if x % batch_size*10 == 0:
    #         logging.info('starting to transform vectors to pca data on batch [{}:{}]'.format(_from, _to))
    #     if len(batch) != 0:
    #         pca_batch = pca.transform(batch)
    #         for i in range(_from,_to):
    #             logging.info('updating sqlitedict values ... {}/{} running time:'.format(i,_to-_from+1,time.time()-start))
    #             try:
    #                 id2pca[ids[i]] = pca_batch[i]
    #             except IndexError:
    #                 logging.warning('probably junk at the end, can ignore!')
    #     if x % batch_size*10 == 0:
    #         logging.info('done pca transformation! | time spent: {} seconds'.format(time.time() - start))
    # id2pca.close()

    id2vector.close()

def play_5():
    id2vector_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite'
    id2vector = SqliteDict(id2vector_path, autocommit=True)
    ctr = 0
    start = time.time()
    vectors = []
    ids = []

    # get vectors into a list
    for id, v in id2vector.iteritems():
        if ctr >= 5000000:
            break
        vectors.append(v)
        ids.append(id)
        if ctr % 100000 == 0:
            logging.info('creating vector and id list! {}/{} | time spent: {} seconds'.format(ctr, 42000000,
                                                                                              time.time() - start))
        ctr += 1

    batch_size = 16
    chunk_size = 1000
    pca = IncrementalPCA(n_components=3, batch_size=batch_size, copy=False)
    logging.info('starting incremental pca!')
    start = time.time()

    #do incremental pca
    for i in range(0, len(vectors)//chunk_size):
        _from = i*chunk_size
        _to = (i+1)*chunk_size

        if i % 1000 == 0:
            logging.info('starting to perform pca on data batch [{}:{}]'.format(_from, _to))
        try:
            pca.partial_fit(vectors[_from : _to])
        except:
            logging.info('empty vector! just ignore this')
        if i % 1000 == 0:
            logging.info('done partial pca for batch [{}:{}]! | time spent: {} seconds'.format(_from, _to, time.time() - start))

    logging.info('done partial pca for all batches! | time spent: {} seconds'.format(time.time() - start))

    # # do transformation and write to database
    # id2pca_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_pca.sqlite'
    # id2pca = SqliteDict(id2pca_path, autocommit=True)
    # for i in range(0, len(vectors) // chunk_size):
    #     _from = i * chunk_size
    #     _to = (i + 1) * chunk_size
    #     if i % 1000 == 0:
    #         logging.info('starting to transform vectors to pca data on batch [{}:{}]'.format(_from, _to))
    #     batch = vectors[_from:_to]
    #     try:
    #         pca_batch = pca.transform(batch)
    #         for i in range(_from, _to):
    #             logging.info('updating sqlitedict values ... {}/{} running time:'.format(i-_from,_to-_from+1,time.time()-start))
    #             try:
    #                 id2pca[ids[i]] = pca_batch[i]
    #             except IndexError:
    #                 logging.warning('probably junk at the end, can ignore!')
    #     except:
    #         logging.info('empty vector! just ignore this')
    #     if i % 1000 == 0:
    #         logging.info('done pca transformation! | time spent: {} seconds'.format(time.time() - start))
    #
    # id2pca.close()

    id2vector.close()

def play_4():
    a = [[[1,2,3],[1,4,5],[5,6,7]],[[1,2,3],[1,4,5],[5,6,7]],[[1,2,3],[1,4,5],[5,6,7]]]
    aa = np.asarray(a)
    bb = np.vstack(aa)
    print(bb)


def play_7():
    id2vector_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite'
    id2vector = SqliteDict(id2vector_path)
    ctr = 0
    example = {}

    for k,v in id2vector.iteritems():
        example[k] = v
        if ctr >= 100:
            break
        ctr += 1

    spark = SparkSession.builder.getOrCreate()
    pddf = pd.DataFrame(example.items())

    df = spark.read.load('id2vector.parquet')
    df.show(1)
    # #empty dataframe
    # df = spark.createDataFrame(pddf)
    # df.show(1)
    # df.write.save('id2vector.parquet')
    # # id2vector.close()

def play_8():
    d = np.zeros((0,3))
    a = [[1,2,3],[4,67,7],[56,536,432],[765,7,54]]
    b = [[4253,7564,768],[5432,453,89],[5324,2314,432],[8657,47,564]]
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.vstack((d,a,b))
    print(c)

def play_9():
    ids = []
    play_range = 100000
    start = time.time()
    for i in range(play_range):
        ids.append(hasher.encode(i))
    log.debug('took {} seconds to encode and append {}'.format(round(time.time()-start,8),play_range))
    print(ids.count())

def play_10():
    print(hasher.decode('dZANN9E'))

def play_11():
    current_len = 10
    total_rows = 4035
    start = time.time()
    ids = map(hasher.encode, list(range(current_len, total_rows)))
    log.debug('{} ids mapped and create in {} seconds!'.format(total_rows-current_len, time.time()-start))

    for i, id in enumerate(ids):
        print(hasher.decode(id))
        if i == 100:
            break

def play_13():
    id2info_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_info.sqlite'
    id2info = SqliteDict(id2info_path, autocommit=True)
    print(id2info[hasher.encode(0)])
    id2info.close(force=True)

def play_14():
    num_items = 40356731
    chunk_size = 100000

def helper_play_15(x):
    return x/2

def play_15():
    a = [2431,564,6785,7896,76543,5432,2134,5463,6875,7895]
    b = ['fdgbh','sfdrae','7u6tyj','qwertf','hetgsd','utyrjk','aSDF','AFSD','DHSFG','nbfd' ]
    c = [76,432,28,84,59,59,672,15,897,35]
    for p in product(a,b,c):
        print(p)

def play_16():
    timestamp = 5648
    print(timestamp%60)
    print((timestamp%60)*60)

def play_17():
    with open('solution.js.txt', 'r') as f:
        jstext = f.readlines()

    for line in jstext:
        racer = py_mini_racer.MiniRacer()
        print(racer.eval(line))

def play_18():
    stuff = range(1000000)
    start =time.time()
    stufflist = {*map(hasher.encode,stuff)}
    print('stuff converted to set in {} seconds!'.format(time.time()-start))

    # start = time.time()
    # stuffset = set(stufflist)
    # print('stuff list converted to set in {} seconds!'.format(time.time() - start))

def play_19():
    print(2**14)

def play_20():
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    model.save('sbert_fast_model')

def play_21():
    model_fast = SentenceTransformer('sbert_fast_model')
    query = 'it is important to drink water regularly'

    start = time.time()
    vector = model_fast.encode(query)
    print('faster model: query encoded in %.4f seconds!' % (time.time() - start))

def play_22():
    model_now = SentenceTransformer('snt_tsfm_model')
    query = 'it is important to drink water regularly'

    start = time.time()
    vector = model_now.encode(query)
    print('current model: query encoded in %.4f seconds!' % (time.time() - start))

def play_23(x, d_algo):
    sample_vector = np.random.uniform(low=-1,high=1,size=(768,))
    vectors = np.random.uniform(low=-1,high=1,size=(x*1000,768))

    start = time.time()
    result = list(map(d_algo, [sample_vector]*len(vectors), vectors))
    ids = list(range(x*1000))
    sorted(dict(zip(ids, result)).items())
    print('distance calculation finished in %.4f seconds from %d members using %s algorithm!' % (time.time()-start,x*1000, d_algo))
    gc.collect()

def ddd(x, y):
    return sum((x-y) ** 2)

def play_23_2(x):
    sample_vector = np.random.uniform(low=-1, high=1, size=(768,))
    vectors = np.random.uniform(low=-1, high=1, size=(x * 1000, 768))

    start = time.time()
    d = [(sample_vector-v)**2 for v in vectors]
    d_s = sorted(d)
    print('distance calculated no mapping in %.4f seconds!' % (time.time()-start))

def play_23_1():
    test = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    d_algorithms = [euclidean, braycurtis, canberra, chebyshev, cityblock, correlation, cosine, jensenshannon,
                    minkowski, sqeuclidean, ddd, fdcityblock, fdsqeuclidean, fdeuclidean,
                    fdcosine, fdhamming]
    for alg in d_algorithms:
        play_23(30, alg)

def play_24():
    example = {'c00':{'test':[123,65743,678,756,5324,2341,4563,6758,8796,6532,4231,52341],'test2':'deqweff'},
               'c01':{'test':[6543,5432,3241,5342,5467,7563,6543,5432,45231,2431,53624,6543],'test2':'ddfgef'},
               'c02':{'test':[1654,5432,42351,536,567,6785,57684,6457,5342,4231,3542,546,876],'test2':'detyuikf'}}

def play_25_helper(x, y):
    return x + y

def play_25():
    a = [4213,3245,4563,5674,5784,654,543,5674,789,985,785,654,543,4321,45,657,756]
    b = [1]*len(a)
    mapp = map(play_25_helper, a, b)
    print(list(mapp))

def play_26():
    a = [4213, 3245, 4563, 5674, 5784, 654, 543, 5674, 789, 985, 785, 654, 543, 4321, 45, 657, 756]
    print([a]*10)

def play_27():
    print(time.time())

def play_28():
    a = [324,453,56,67,87,987,7,6,453,423,54,65,76,876,8,6,612,876,9876,63,34]
    b = [(i,x) for i,x in enumerate(a)]
    print(b[:,1])

def play_29():
    arr = []
    dict = {}

    start_list = list(range(10000))
    start = time.time()
    for x in start_list:
        arr.append(x)
    print('appending finished in %.4f seconds!' % (time.time()-start))

    start = time.time()
    for i, x in enumerate(start_list):
        dict[i] = x
    print('dictinoary assignment finished in %.4f seconds!' % (time.time() - start))

def play_30_1(key):
    id2vector = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite')
    id2vector_medium = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id2vector_medium.sqlite', autocommit=True)
    id2vector_medium[key] = id2vector[key]

def play_30():
    id2vector = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite')
    id2vector_medium = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id2vector_medium.sqlite', autocommit=True)
    finish = 22630575
    max = 40356731
    start = time.time()
    valid_ids = list(map(hasher.encode, list(range(10000))))
    print('ids encoded in %.4f seconds!' % (time.time()-start))
    start = time.time()

    # init multiprocessing module
    num_cores = multiprocessing.cpu_count()
    p = PathosPool(num_cores)
    p.map(play_30_1, valid_ids)
    print('%d vectors inserted in %.4f seconds!' % (len(valid_ids), time.time()-start))

    id2vector.close()
    id2vector_medium.close()

def play_31():
    id2vector = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite')
    id2info = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_info.sqlite')
    key = hasher.encode(1234)
    print(id2vector[key])
    print(id2info[key])

    id2vector.close()
    id2info.close()

def keyerror_patch_retrieve_vectors():
    id2vector = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite')
    # id2vector_backup = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id2vector_medium.sqlite')
    id2info = SqliteDict('/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_info.sqlite')
    aa = hasher.encode(22255001)
    bb = hasher.encode(22355001)

    start = time.time()
    ids_to_insert = list(map(hasher.encode, list(range(22255001, 22355001))))
    print('ids generated in %.4f seconds!' % (time.time() - start))

    model = SentenceTransformer('snt_tsfm_model/')
    model.encode('random query!')  # avoid cold start
    texts = []

    start = time.time()
    for i, id in enumerate(ids_to_insert):
        if i % 100 == 0:
            print('working on {}/{}'.format(i, len(ids_to_insert)))
        text = id2info[id]['text']
        texts.append(text)
    print('created array of text in %.4f seconds!' % (time.time() - start))

    start = time.time()
    vectors = model.encode(texts)
    print('encoded chunk of vectors in %.4f seconds!' % (time.time() - start))

    with open('vectors_patch.pkl', 'wb') as f:
        pickle.dump(vectors, f)

    id2vector.close()
    id2info.close()


def insert_vectors():
    id2vector_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_vector.sqlite'
    id2info_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/id_to_info.sqlite'
    id2vector = SqliteDict(id2vector_path, autocommit=True)
    id2info = SqliteDict(id2info_path)
    start = time.time()
    ids_to_insert = list(map(hasher.encode, list(range(22255001, 22355001))))

    with open('vectors_patch.pkl', 'rb') as f:
        vectors = pickle.load(f)

    for i, id in enumerate(ids_to_insert):
        id2vector[id] = vectors[i]
        print('inserted vector %d/%d, running time: %.4f seconds' % (i, len(ids_to_insert), time.time() - start))

    id2vector.close()
    id2info.close()

def prepare_vector_ids():
    lv2_clusters_path = '/media/chanwoo/CChoi/Work/year_5/ttds3_script_vectors/double_clusters_medium.sqlite'
    lv2_clusters = SqliteDict(lv2_clusters_path)

    lv2_keys = []

    for key in lv2_clusters.iterkeys():
        lv2_keys.append(key)

    with open('lv2_keys.pkl', 'wb') as f:
        pickle.dump(lv2_keys, f)

    lv2_clusters.close()

def play_32():
    key = hasher.decode('7DX102Q')
    print(key)

if __name__ == '__main__':
    # play_3()
    # play_4()
    # play_5()
    # play_6()
    # play_7()
    # play_23(5)
    # play_23_1()
    # play_23_2(30)
    # play_27()
    # play_28()
    # play_29()
    # play_30()
    # keyerror_patch_retrieve_vectors()
    # insert_vectors()
    # prepare_vector_ids()
    play_32()