""" This file won't be useful; the geolocation portions of the Twitter API require elevated permissions, and
these were denied to us through manual review. """
"""
NOTE: I got some help from this page: https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a
in paritcular, obtained part of teh get_auth_headers and get_bearer_token and geT_with_auth functions from that link
"""
from Utility import *
import urllib.parse
import requests


def get_bearer_token():
    token_file = open(PATH_TO_TWITTER_API_BEARER_TOKEN)
    return token_file.read()


def get_auth_headers():
    headers = {"Authorization": "Bearer {}".format(get_bearer_token())}
    return headers

def get_with_auth(url, params=None):
    response = None
    if params is None:
        response = requests.request(url, headers=get_auth_headers())
    else:
        response = requests.request(url, headers=get_auth_headers(), params=params)

    if response.status_code != 200:
        raise Exception(f"Error getting url {url}; got status code {response.status_code} and error {response.text}")

    return response.json()


def get_query(keywords, latitude, longitude):
    """

    :param keywords (list of strings): a list of the keywords to query for
    :param latitude (float):
    :param longitude (float): the lat and long are the lat/long of the location around which we want to query.
    :return: the URL escaped query string for querying the twitter API
    """
    SCHEME = "https"
    ENDPOINT = "api.twitter.com/2/tweets/search/recent"
    pass

def get_tweet_for_id(tweet_id):
    """

    :param tweet_id:
    :return: the text for the tweet corresponding to the given tweet id.
    """
    #SCHEME = "https"
    #ENDPOINT = f"api.twitter.com/2/tweets/{tweet_id}"
    ENDPOINT = f"https://api.twitter.com/2/tweets/{tweet_id}"
    response = get_with_auth(ENDPOINT)
    return response['data']['text']



#https://?query=-has%3Ageo%20(povery%20ORwealth)