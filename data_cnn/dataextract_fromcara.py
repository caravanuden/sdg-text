import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, classification_report
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
import seaborn as sns
import time

from utils.get_data_loader import SustainBenchTextDataset


############## criterion
THRESHOLD_DICT = {'asset_index': 0, 'sanitation_index': 3, 'water_index': 3, 'women_edu': 5}
TARGETS = ['asset_index', 'sanitation_index', 'water_index', 'women_edu']
key = 'DHSID_EA'
maxlen = 1000


target = 'asset_index'
labels_df = pd.read_csv('../data/dhs_final_labels.csv')
sentences_df = []

relevant_sentences_dir = '../data/relevant_sentences'
for country in os.listdir(relevant_sentences_dir):
    sentences_for_country_df = pd.read_csv(
        os.path.join(relevant_sentences_dir, country, target, 'relevant_sentences.csv'))
    sentences_for_country_df = sentences_for_country_df.merge(labels_df, on='DHSID_EA')[
        ['DHSID_EA', 'relevance_score', 'most_relevant_sentences', target]]
    sentences_for_country_df = sentences_for_country_df.sort_values(by=['DHSID_EA', 'relevance_score'])
    sentences_for_country_df = sentences_for_country_df.groupby('DHSID_EA')['most_relevant_sentences'].apply(
        list).reset_index(name='most_relevant_sentences_per_loc')
    sentences_for_country_df['most_relevant_sentences'] = sentences_for_country_df[
        'most_relevant_sentences_per_loc'].apply(lambda x: ' '.join(x))
    # the length of most_relevant_sentences in words
    sentences_for_country_df['len_most_relevant_sentences'] = sentences_for_country_df['most_relevant_sentences'].apply(
        lambda x: len(x.split(' ')))
    sentences_for_country_df['country_name'] = country
    sentences_for_country_df = sentences_for_country_df[
        ['DHSID_EA', 'country_name', 'most_relevant_sentences', 'len_most_relevant_sentences']]
    sentences_df.append(sentences_for_country_df)

sentences_df = pd.concat(sentences_df, axis=0)
sentences_df.head()


data_and_labels = sentences_df.merge( labels_df, on=key, how="left")[[key, target,  'most_relevant_sentences']]
for i, target in enumerate(TARGETS):
    data_and_labels = sentences_df.merge(labels_df, on=key, how="left")[[key, target, 'most_relevant_sentences', 'len_most_relevant_sentences']]
    data_and_labels = data_and_labels[data_and_labels['len_most_relevant_sentences'] < maxlen]
    # data = data_and_labels.merge(labels_df, on=key, how="left")[['most_relevant_sentences']]
    # labels = data_and_labels.merge(labels_df, on=key, how="left")[[target]]
    labels = data_and_labels[target]
    labels = labels < THRESHOLD_DICT[target]
    labels_save = labels.astype(int).iloc[:]
    data_save = data_and_labels.iloc[:,2]


    data_save = data_save.str.replace('"', '')
    labels_save.to_csv(target+'_label.txt', header=None, index=None, sep=';')
    data_save.to_csv(target+'_data.txt', header=None, index=None, sep=';')

