import glob
from collections import defaultdict

import pandas as pd
from selenium.webdriver.support.wait import WebDriverWait
from youtube_transcript_api import YouTubeTranscriptApi
import itertools
import pickle
import requests
from pathlib import Path
import json
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import pyperclip
import geckodriver_autoinstaller

class VidCrawler():

    def init_nd_dict(self):
        return defaultdict(lambda: defaultdict(dict))

    def load_files(self, path):
        with open(path) as f:
            temp = f.read().split('\n')
            loaded = []
            [loaded.append(x) for x in temp if x != '']
            return loaded

    def go_crawl(self, url):
        video_id = ''
        ids = []
        # options = Options()
        fox_profile = '/home/chanwoo/.mozilla/firefox/6aet0uo5.default-release'
        profile = webdriver.FirefoxProfile(
            fox_profile)
        profile.set_preference("dom.webdriver.enabled", False)
        profile.set_preference('useAutomationExtension', False)
        profile.update_preferences()
        desired = DesiredCapabilities.FIREFOX

        driver = webdriver.Firefox(firefox_profile=profile,
                                   desired_capabilities=desired)

        # driver = webdriver.Firefox(options=options)
        driver.get(url)
        driver.maximize_window()
        time.sleep(3)
        for i in range(5000): #5000 is just random number
            try:
                # this is needed when the user is not authenticated
                # # agree to PP statement
                # driver.find_element_by_css_selector('ytd-button-renderer.style-scope:nth-child(2) > a:nth-child(1) > '
                #                                     'tp-yt-paper-button:nth-child(1) > yt-formatted-string:nth-child(1)').click()

                #click share
                driver.find_element_by_css_selector(
                    'ytd-button-renderer.ytd-menu-renderer:nth-child(3) > a:nth-child(1) > yt-formatted-string:nth-child(2)').click()

                #copy video id to clipboard by clicking "copy"
                driver.find_element_by_css_selector('yt-formatted-string.yt-button-renderer').click()

                #paste the video url from clipboard
                url = pyperclip.paste()
                video_id = str(url)[17:]
                print(video_id)

                with open('video_ids.txt', 'a') as f:
                    f.write(str(video_id)+'\n')

                #escape share view
                driver.find_element_by_css_selector('yt-icon.ytd-unified-share-panel-renderer').click()

                #move to next video
                driver.find_element_by_css_selector('.ytp-next-button').click()

                #wait for the page to be updated
                time.sleep(1)

            except:     #if too quick to get the video id then just refresh - this happens very rarely
                #refresh
                driver.refresh()
                time.sleep(5)

    """use youtube api to get texts (in json) including loads of video ids 
        -> ids to be extracted using the method extract_vid_ids_from_json"""
    def get_text_data_using_api(self):
        api_key = 'AIzaSyC01e0xgVdQGzDKf0eCuj6c7VjlOAquKiI'     #my api key for Google Auth
        api_cost_counter = 0
        channel_ids = self.load_files('channel_ids.txt')
        for i, channel_id in enumerate(channel_ids):
            ########################### API call! be cautious or you'll get ripped off ###############################
            print('extracting video ids from channel id [{}] ... channel num {}/{}'.format(channel_id, i, len(channel_ids)))

            # get upload id first; with this id we can get the entire list of videos from a channel
            data = requests.get('https://www.googleapis.com/youtube/v3/channels?id={}&'
                                'key={}&part=contentDetails'.format(channel_id, api_key))
            api_cost_counter += 1
            upload_id = data.json()['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            # print(upload_id)

            # now we get video ids using the upload id, start retrieving from first page
            response = requests.get('https://www.googleapis.com/youtube/v3/playlistItems?playlistId={}&'
                                    'key={}&part=snippet&maxResults=50'.format(upload_id, api_key))
            api_cost_counter += 1
            # save to avoid unnecessary later api calls
            with open('youtube_api_data/{}_{}.json'.format(channel_id,'0'), 'w') as f:
                json.dump(response.json(), f)

            # now get video ids from next page until we reach the last page
            while True:
                try:
                    next_page_token = response.json()['nextPageToken']
                    print(next_page_token)
                    response = requests.get('https://www.googleapis.com/youtube/v3/playlistItems?playlistId={}&'
                                            'key={}&part=snippet&maxResults=50&pageToken={}'.format(upload_id, api_key, next_page_token))
                    api_cost_counter += 1
                    # save to avoid unnecessary api calls
                    with open('youtube_api_data/{}_{}.json'.format(channel_id, next_page_token), 'w') as f:
                        json.dump(response.json(), f)
                except KeyError:
                    break
            ##########################################################################################################
            print("Youtube API cost: {} units".format(api_cost_counter))

    def extract_vid_ids_from_json(self):
        dir = 'youtube_api_data'
        paths = Path(dir).rglob('*.json')
        num_paths = len(glob.glob('youtube_api_data/*'))
        counter = 1
        for path in paths:
            print('extracting video ids from channels ... {}/{}'.format(counter, num_paths))
            with open(str(path), 'r') as f:
                # api_data = json.load(f)
                try:
                    api_data = json.load(f)
                    for i in range(len(api_data['items'])):
                        with open('video_ids.txt', 'a') as f:
                            vid = api_data['items'][i]['snippet']['resourceId']['videoId']
                            print(vid)
                            f.write(vid+'\n')
                except:
                    print("failed to extract ids from: {}".format(str(path)))
            counter += 1

    def validate_ids(self):
        with open('video_ids.txt') as f:
            ids = f.read().split('\n')

        ids_no_dup = []
        for i, id in enumerate(ids):
            if i % 1000 == 0:
                print('removing duplicate ids from the list ... {}%'.format(round(i/len(ids)*100,2)))
            if id not in ids_no_dup:
                ids_no_dup.append(id)

        # [ids_no_dup.append(x) for x in ids if x not in ids_no_dup]

        #write the ids to a txt file
        with open('video_ids_uniq.txt', 'a') as f:
            for id in ids_no_dup:
                f.write(str(id)+'\n')

        #final check if there any duplicates
        with open('video_ids_uniq.txt', 'r') as f:
            vids = f.read().split('\t')
        if len(vids) == len(set(vids)):
            print("Duplicates are removed successfully!")
        else:
            print("There are still duplicates in the list!")

    def get_scripts(self, resume=0):
        with open('video_ids_uniq.txt', 'r') as f:
            ids = f.read().split('\n')

        #initiate csv filie for storing the scripts
        try:        # if file contains nothing then add column labels
            with open('scripts.csv', 'r+') as f:
                if f.read() == '':  # if the file exists but empty then put entercolumn labels
                    f.write('"video_id","text","timestamp","duration"\n')
                #else do nothing
        except:     # if the file does not exist
            with open('scripts.csv', 'w') as f:
                f.write('"video_id","text","timestamp","duration"\n')

        for i, id in enumerate(ids):
            print('finding scripts for video [{}] ... {}/{}'.format(id, i, len(ids)))
            if i < resume:
                continue
            else:
                try:
                    script_data = YouTubeTranscriptApi.get_transcript(id)

                    for j in range(len(script_data)):

                        ############# data to store #############
                        vidid = id
                        text = script_data[j]['text'].replace('\n',' ').replace('"',' ') # remove annoying \ns that screw up the csv format
                        timestamp = script_data[j]['start']
                        duration = script_data[j]['duration']
                        ########################################

                        with open('scripts.csv', 'a') as f: # save the data into a csv file (splittable by \n)
                            f.write('"{}","{}","{}","{}"\n'.format(vidid, text, timestamp, duration))
                except:
                    print("[Warning] Failed to retrieve transcripts, this channel does not allow auto-generated transcripts!")

    #remove double double quotes to make for format work
    def validate_scripts(self):
        with open('scripts.csv', 'r') as f:
            original_scripts = f.readlines()

        #remove any other annoying double quotation marks
        with open('scripts_reformatted.csv', 'a') as g:
            for i, line in enumerate(original_scripts):
                print('reformatting lines ... {}/{}'.format(i, len(original_scripts)))
                new_line = '"' + line.replace('","','qxqxq').replace('"',' ').replace('qxqxq','","')[1:-2] + '"\n'
                g.write(new_line)


    def load_scripts(self):
        # scripts = pd.read_csv('scripts.csv')

        try:
            scripts = pd.read_csv('scripts.csv')
            print(len(scripts))
        except:
            raise SyntaxError("load failed! the file is not in valid csv format!")

        print("Script load to json test passed!\n")

class ChannelCrawler():

    def __init__(self):
        self.channels_list = self.load_list('channel_urls.txt')

    def load_list(self, path):
        loaded = []
        with open(path) as f:
            temp = f.read().split('\n')
            for item in temp:
                if item != '':
                    loaded.append(item)
        return loaded

    def get_channels_list(self, mode):
        options = Options()
        driver = webdriver.Firefox(options=options)
        if mode == 'feedspot':
            url = 'https://blog.feedspot.com/educational_youtube_channels/'
            driver.get(url)
            elems = driver.find_elements_by_xpath("//a[@href]")
            with open('channel_urls.txt', 'a') as f:
                for elem in elems:
                    url = elem.get_attribute("href")
                    if 'youtube.com/user/' in str(url):
                        f.write(url+'\n')

        elif mode == 'socialblade':
            url = 'https://socialblade.com/youtube/top/category/tech/mostsubscribed'
            driver.get(url)
            driver.maximize_window()

            #wait for everything to be loaded
            time.sleep(2)

            #find and switch to iframe (there is an iframe for the Privacy Notice window)
            iframe = driver.find_element_by_xpath('//*[@id="sp_message_iframe_541073"]')
            driver.switch_to.frame(iframe)

            #click Accept button and move on
            WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.message-component:nth-child(2)"))).click()

            #for now just refresh to go back to the original iframe - FIX IT LATER!
            driver.refresh()

            elems = driver.find_elements_by_xpath("//a[@href]")
            with open('channel_urls.txt', 'a') as f:
                for elem in elems:
                    url = elem.get_attribute("href")        #find channel urls
                    if 'youtube/c/' in str(url):
                        url_searchable_form = url.replace('/c/','/user/')
                        if '%' not in url_searchable_form:
                            f.write(url_searchable_form +'\n')   #replace /c/ with /user/ to make it searchable
                        print(url_searchable_form)

        elif mode == 'manual':
            channel_urls = [
                #according to: https://www.inc.com/jessica-stillman/youtube-videos-science-education-google.html
                'https://www.youtube.com/c/inanutshell',
                'https://www.youtube.com/c/rsaorg',
                'https://www.youtube.com/c/talksatgoogle',
                'https://www.youtube.com/c/misterwootube',
                'https://www.youtube.com/c/veritasium',
                'https://www.youtube.com/c/smartereveryday',
                'https://www.youtube.com/c/CoinBureau',
                'https://www.youtube.com/c/AndrewHubermanLab',
                'https://www.youtube.com/c/aliabdaal',
                'https://www.youtube.com/c/Vox',

                #according to: https://www.uopeople.edu/blog/best-educational-youtube-channels-for-college-students/
                'https://www.youtube.com/greymatter',
                'https://www.youtube.com/results?search_query=Khan+Academy',
                'https://www.youtube.com/c/AsapSCIENCE',
                'https://www.youtube.com/c/CommonSenseEducators',
                'https://www.youtube.com/c/edutopia',
                'https://www.youtube.com/c/bigthink',
                'https://www.youtube.com/c/NatGeo',
                'https://www.youtube.com/user/expertvillage',
                'https://www.youtube.com/c/minutephysics',
                'https://www.youtube.com/c/SciShow',
                'https://www.youtube.com/c/numberphile',
                'https://www.youtube.com/c/badastronomy',
                'https://www.youtube.com/c/sickscience',
                'https://www.youtube.com/c/lifenoggin',
                'https://www.youtube.com/user/Computerphile',
                'https://www.youtube.com/c/AppliedScience',
                'https://www.youtube.com/c/Thomasfrank',

                #according to: https://thenerddaily.com/6-informative-youtube-channels/
                'https://www.youtube.com/c/CrimsonEducation',
                'https://www.youtube.com/c/SuperCarlinBrothers'


            ]
            with open('channel_urls.txt', 'a') as f:
                for url in channel_urls:
                    f.write(url+'\n')
        else:
            raise ValueError('wrong search mode! We now only support Feedspot and Socialblade.')

    def get_channel_ids(self, resume_from=0):
        options = Options()
        channel_to_id_url = 'https://commentpicker.com/youtube-channel-id.php'
        driver = webdriver.Firefox(options=options)
        driver.get(channel_to_id_url)
        driver.maximize_window()
        time.sleep(2)

        #accept cookies
        driver.find_element_by_css_selector('#ez-accept-all').click()

        channel_ids = {}

        with open('channel_ids.txt', 'a') as f:
            for i, ch in enumerate(self.channels_list):
                if resume_from > 0 and i < resume_from: continue
                driver.find_element_by_css_selector('#js-youtube-url').send_keys(ch)
                driver.find_element_by_css_selector('div.ezmob-footer:nth-child(2) > span:nth-child(2)').click() #remove the annoying white box
                driver.find_element_by_css_selector('#get-channel-id').click()          #click button to get the id
                ch_id = driver.find_element_by_css_selector('#js-results-id').text      #get the channel id text
                channel_ids[ch] = ch_id
                print('fetching channeling id ... {}/{}'.format(i, len(self.channels_list)))

                if ch_id == '-':
                    while ch_id == '-':
                        #if failed to get the channel id, refesh and wait to give it another chance
                        driver.refresh()
                        time.sleep(2)
                        print('channel id retrieval failed! Having another try ...')
                        #another try
                        driver.find_element_by_css_selector('#js-youtube-url').send_keys(ch)
                        driver.find_element_by_css_selector(
                            'div.ezmob-footer:nth-child(2) > span:nth-child(2)').click()  # remove the annoying white box
                        driver.find_element_by_css_selector('#get-channel-id').click()  # click button to get the id
                        time.sleep(2)
                        ch_id = driver.find_element_by_css_selector('#js-results-id').text  # get the channel id text
                        channel_ids[ch] = ch_id
                    f.write(str(ch_id) + '\n')  # write up the result to file
                    print(ch_id)
                    print("channel id fetch success! moving on ...")
                    driver.refresh()
                    time.sleep(1)
                else:
                    print(ch_id)
                    f.write(str(ch_id) + '\n')                          #write up the result to file

                    # refresh for another input
                    driver.refresh()

                # url_box = driver.find_element_by_css_selector('.search-wrapper > ul:nth-child(3) > li:nth-child(3) > a:nth-child(1)')
                # driver.execute_script("arguments[0].scrollIntoView();", url_box)

    "todo"
    def go_search(self, url):
        video_id = ''
        ids = []
        # options = Options()
        fox_profile = '/home/chanwoo/.mozilla/firefox/6aet0uo5.default-release'
        profile = webdriver.FirefoxProfile(
            fox_profile)
        profile.set_preference("dom.webdriver.enabled", False)
        profile.set_preference('useAutomationExtension', False)
        profile.update_preferences()
        desired = DesiredCapabilities.FIREFOX

        driver = webdriver.Firefox(firefox_profile=profile,
                                   desired_capabilities=desired)

        # driver = webdriver.Firefox(options=options)
        driver.get(url)
        driver.maximize_window()
        time.sleep(3)
        for i in range(5000): #5000 is just random number
            try:
                # this is needed when the user is not authenticated
                # # agree to PP statement
                # driver.find_element_by_css_selector('ytd-button-renderer.style-scope:nth-child(2) > a:nth-child(1) > '
                #                                     'tp-yt-paper-button:nth-child(1) > yt-formatted-string:nth-child(1)').click()

                #click share
                driver.find_element_by_css_selector(
                    'ytd-button-renderer.ytd-menu-renderer:nth-child(3) > a:nth-child(1) > yt-formatted-string:nth-child(2)').click()

                #copy video id to clipboard by clicking "copy"
                driver.find_element_by_css_selector('yt-formatted-string.yt-button-renderer').click()

                #paste the video url from clipboard
                url = pyperclip.paste()
                video_id = str(url)[17:]
                print(video_id)

                with open('video_ids.txt', 'a') as f:
                    f.write(str(video_id)+'\n')

                #escape share view
                driver.find_element_by_css_selector('yt-icon.ytd-unified-share-panel-renderer').click()

                #move to next video
                driver.find_element_by_css_selector('.ytp-next-button').click()

                #wait for the page to be updated
                time.sleep(1)

            except:     #if too quick to get the video id then just refresh - this happens very rarely
                #refresh
                driver.refresh()
                time.sleep(5)

"""""""""crawl to get youtube channel ids"""""""""
cc = ChannelCrawler()
modes = ['feedspot', 'socialblade', 'manual']
# cc.get_channels_list(modes[2])    # get informative channel urls
# cc.get_channel_ids()              # get channel ids using the urls
""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""" search url for manual crawling [probably outdated/to-be-removed] """""""""""""""""""
# search_url = 'https://www.youtube.com/watch?v=YmpFd8Ikjtc&list=UUAuUUnT6oDeKwE6v1NGQxug'
# search_url_resume = 'https://www.youtube.com/watch?v=mGbMwP8MDjg&list=UUAuUUnT6oDeKwE6v1NGQxug&index=1288' #in case I have to resume
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""" get video ids using channel ids """""""""""""""""""""""""""""""""""""
vc = VidCrawler()
# vc.go_crawl(search_url)               # manual crawling without using youtube API - probably outdated/to-be-removed
# vc.get_text_data_using_api()          # get data chunk including video ids from youtube api
# vc.extract_vid_ids_from_json()        # extract video ids from the data chunk
# vc.validate_ids()                     # remove duplicate video ids
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""" get scripts using video ids """""""""
# vc.get_scripts(97200)
# vc.validate_scripts()
vc.load_scripts()           #test saved file
""""""""""""""""""""""""""""""""""""""""""""
