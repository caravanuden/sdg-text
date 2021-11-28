""" This file won't be useful; the geolocation portions of the Twitter API require elevated permissions, and
these were denied to us through manual review. """
from Utility import *
import urllib.parse

def get_bearer_token():
    token_file = open(PATH_TO_TWITTER_API_BEARER_TOKEN)
    return token_file.read()


def get_auth_headers():
    headers = {"Authorization": "Bearer {}".format(get_bearer_token())}
    return headers


def get_query(keywords, latitude, longitude):
    """

    :param keywords (list of strings): a list of the keywords to query for
    :param latitude (float):
    :param longitude (float): the lat and long are the lat/long of the location around which we want to query.
    :return: the URL escaped query string for querying the twitter API
    """
    SCHEME = "https"
    ENDPOINT = "api.twitter.com/2/tweets/search/recent"


#https://?query=-has%3Ageo%20(povery%20ORwealth)