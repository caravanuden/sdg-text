
import csv
import numpy as np
import nltk
# nltk.download('words')
import re
import pandas as pd
import glob
import os

# os.chdir("C:\Users\Wonkyung/Do\PycharmProjects\pythonProject")

def load_relevant_sentences(path, type):
    """
    Convert relevant sentence data into one concatenated files
    :return:
    """
    extension = 'csv'
    pathname = path+'*/'+type+'/*.{}'.format(extension)
    # pathname = 'relevant_sentences/*.csv'
    print(pathname)

    all_filenames = [i for i in glob.glob(pathname)]
    print(all_filenames)
    print(len(all_filenames))
    # combine all files in the list
    combined_csv = pd.read_csv(all_filenames[0])
    combined_csv = combined_csv.iloc[:,1:]

    for f in all_filenames[1:3]:
        label = pd.read_csv(f)
        label.iloc[:,1:]
        # print(label)
        pd.concat([combined_csv, label],axis=0)


    combined_csv.to_csv("combined.csv", index=False, encoding='utf-8-sig')

    index = combined_csv.iloc[:,0]
    index = index.astype(int)

    # convert index in two ways for classification - 0 and 1
    if type == 'asset_index':
        index = index.replace([3,4,6],2)
        index = index- 1

    data = combined_csv.iloc[:,1]

    data = data.str.replace('"', '')
    index.to_csv(type+'label.txt', header=None, index=None, sep=';')
    data.to_csv(type+'data.txt', header=None, index=None, sep=';')
    return pd.read_csv('combined.csv')


def load_text(path):
    """Load text data, lowercase text and save to a list."""
    with open(path, 'rb') as f:
        texts = []
        for line in f:
            texts.append(line.decode(errors='ignore').lower().strip())
    return texts

if __name__ == '__main__':

    combined = load_relevant_sentences('relevant_sentences/','asset_index')
    combined = load_relevant_sentences('relevant_sentences/','asset_index')
    combined = load_relevant_sentences('relevant_sentences/','asset_index')

    # a = load_text('data.txt')
    # print(a[0:3])
    # pos_text = load_text('rt-polarity.pos')
    # print(pos_text[0:3])


######### asset index ########
# index.value_counts()
# 1.0    3622
# 2.0     504
# 3.0     120
# 4.0      42
# 6.0       2
# Name: relevance_score, dtype: int64
# So let's combine from 2 to 6

# 1.0    3622
# 2.0     668
# Name: relevance_score, dtype: int64