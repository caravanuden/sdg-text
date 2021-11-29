"""Note: large portions of this code file are publicly available under a repo I created for a prior research project (called Towards-Understanding-Gender-Bias-In-Neural-Relation-Extraction
The functions in this file which are also in that file are functions I use in all my Python projects; hence, I
included those functions below as well."""

import json
import os
from pathlib import Path

PATH_TO_DHS_LABELS = "./dhs_labels/dhs_final_labels.csv"
PATH_TO_WIKIPEDIA_OUTPUTS = "./wiki_outputs"
PATH_TO_GDELT_OUTPUTS = "./gdelt_outputs"
PATH_TO_PREPROCESSED_DOC2VEC_INPUTS = "./doc2vec_inputs"
PATH_TO_DOC2VEC_MODEL = "./doc2vec_model"
PATH_TO_TWITTER_API_KEYS = "./twitter_api_keys"
PATH_TO_TWITTER_API_BEARER_TOKEN = os.path.join(PATH_TO_TWITTER_API_KEYS, "bearer_token.txt")
PATH_TO_TWITTER_OUTPUTS = "./twitter_outputs"


def readFromJsonFile(infile_path):
    with open(infile_path, 'r') as infile:
        return json.load(infile)
def writeToJsonFile(data, outfile_path, prettify=False):
    Path('/'.join(outfile_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True) # create directory if necessary
    with open(outfile_path, 'w') as outfile:
        if(prettify):
            json.dump(data, outfile, indent=4, sort_keys=True)
        else:
            json.dump(data, outfile)

def writeToAnyFile(data, outfile_path):
    Path('/'.join(outfile_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True) # create directory if necessary
    with open(outfile_path, 'w') as f:
        f.write(data)


def writeJsonToDirectoryAndFile(json_object, path, final_filename='object.json'):
    '''
    :param json_object: the object we want to write to the directory structure
    :param path: the path of the parent dir to which we want the file writing to start
    :param final_filename: the name of the file with data at the end of the writing
    :return: nothing, but write the json object to the directory structure
    '''
    for k, v in json_object.items():
        new_path = os.path.join(path, k)
        if isinstance(v, dict):
            writeJsonToDirectoryAndFile(v, new_path, final_filename)
        else:
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            file_path = os.path.join(new_path, final_filename)
            writeToJsonFile(v, file_path)
            return
def readFromDirectoryAndFileToJson(dir_path, json_object=dict(), take_key=False):
    '''
    :param dir_path: the original parent directory from which we will create the json object
    :param json_object: for the internal recursion (this is the json dictinary object which gets built as the fuction recurs)
    :param take_key:  for the internal recursion (this basically just prevents us from using the original parent directory as the key)
    :return: a json object (it's a dictionary where any directory becomes a key in the object, and any feil at the end of that directory
    crawl becomes a value in the dictionary
    '''
    key = dir_path.split('/')[-1]
    if key == '':
        key = dir_path.split('/')[-2]
    if key not in json_object and take_key:
        json_object[key] = dict()
    for item in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, item)):
            if take_key:
                readFromDirectoryAndFileToJson(os.path.join(dir_path, item), json_object[key], True)
            else:
                readFromDirectoryAndFileToJson(os.path.join(dir_path, item), json_object, True)
        else:
            # it's a file
            if key not in json_object or isinstance(json_object[key], dict):
                json_object[key] = list()
            json_object[key].append(readFromJsonFile(os.path.join(dir_path, item)))


    return json_object

def makeDirs(path, is_file=False):
    '''

    :param path: the path to create
    :param is_file: if True, then we create the dirs for the entire path ASIDE from the file
    (i.e., a/b/c.txt creates dirs a/b/). If False, then it creates the full path (e.g. a/b/c/)
    :return: create dirs.
    NOTE: if dir already exists, that's OK! this won't change anything then.
    '''
    if is_file:
        Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)  # create directory if necessary
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

