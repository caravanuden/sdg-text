from dhs_reader import DHSReader
from Utility import *
import numpy as np
import os
import json
from scrape_twitter import *

def clean_tweet(tweet_text):
    """

    :param tweet_text: the tweet text
    :return: cleaned tweet text
    """

    # temporary
    return tweet_text

def read_in_tweet_geolocation_data():
    """

    :return: (tuple(string, float, float)): yield a tuple of (tweet_id, latitude, longtidue) associated with teh tweet
    NOTE: the tweet files SAY that they're JSON, but in reality they have a single JSON dictionary object per line (so the
     full file isn't valid jSON, but it's really easy to read it in line by line)
    """
    file_names = ["tc2014.json", "tc2015.json"]
    for file_name in file_names:
        full_file_path = os.path.join("geolocated_tweets", "tweet-geolocation-5m", file_name)

        f = open(full_file_path, 'r')
        for line in f.readlines():
            tweet_json = json.loads(line)
            yield (tweet_json['tweet_id'], float(tweet_json['location']['latitude']), float(tweet_json['location']['longitude']))
        f.close()



def map_tweets_to_dhs_id():
    """

    :return: a dict mapping tweet_ids to dhs_ids (it'll map the dhs id to the tweet id when that tweet_id has a
     non-None corresponding closest dhs row according to dhs reader)
    """
    tweet_id_to_dhs_id = dict()
    dhs_reader = DHSReader()
    for (tweet_id, latitude, longitude) in read_in_tweet_geolocation_data():
        closest_dhs_id = dhs_reader.get_closest_dhs_ids_to_lat_lon(latitude, longitude)
        if closest_dhs_id is not None:
            tweet_id_to_dhs_id[tweet_id] = closest_dhs_id

    return tweet_id_to_dhs_id


def get_all_relevant_tweets():
    """

    :return: (list of dicts) a list of dictionaries, with each dictionary being structured as
     {
        "clean_text": tweet text
        "tag": DHS_ID-SOME_NUMBER-twitter
     }
    """
    outputs = list()
    output_file_counter = 0

    dhs_id_counters = dict()
    tweet_id_to_dhs_id = map_tweets_to_dhs_id()
    for tweet_id in tweet_id_to_dhs_id:
        tweet_text = get_tweet_for_id(tweet_id)
        tweet_text = clean_tweet(tweet_text)

        dhs_id = tweet_id_to_dhs_id[tweet_id]
        if dhs_id in dhs_id_counters:

            dhs_id_counters[dhs_id] += 1
        else:
            dhs_id_counters[dhs_id] = 0
        curr_num = dhs_id_counters[dhs_id]

        outputs.append({
            "clean_text": tweet_text,
            "tag": f"{dhs_id}-{curr_num}-twitter"
        })

        if len(outputs > 1000):
            writeToJsonFile(outputs, os.path.join(PATH_TO_TWITTER_OUTPUTS,f"tweets_{output_file_counter}"))
            output_file_counter += 1
            outputs = list()

    writeToJsonFile(outputs, os.path.join(PATH_TO_TWITTER_OUTPUTS, f"tweets_{output_file_counter+1}"))


if __name__ == '__main__':
    get_all_relevant_tweets()

