from copy import deepcopy
import gensim.downloader as gensim_api
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import string
import re
from keywords import CHARMANDER_KEYWORDS_DICT


def _process_keyword(keyword, lemmatize_model=None):
    keyword = keyword.strip("-").lower()
    if lemmatize_model is not None:
        keyword = lemmatize_model.lemmatize(keyword)
    return keyword


def get_related_keywords(
    starter_keywords_dict, lemmatize_model, word_similarity_model, n=25
):
    """
    Using pretrained gensim model, adds related keywords based off the
    starter keywords provided for each target in `starter_keywords_dict`.

    Args:
        `starter_keywords_dict`: Dict mapping target to starter keywords. Example:
            {'sanitation_index':['sanitation'], 'women_edu':['woman', 'education']}
        `n`: How many related words to return

    Returns:
        Dict mapping target to starter keywords, expanded with `n` related keywords for each entry.
    """
    keywords_dict = {}

    for target, starter_keywords in starter_keywords_dict.items():
        keywords_dict[target] = deepcopy(starter_keywords)

        # get keywords most similar to both
        similar_keywords = word_similarity_model.most_similar(starter_keywords, topn=n)
        keywords_dict[target] += [keyword_tup[0] for keyword_tup in similar_keywords]

        # get keywords most similar to each starter keyword
        for keyword in starter_keywords:
            similar_keywords = word_similarity_model.most_similar(keyword, topn=n)
            keywords_dict[target] += [
                keyword_tup[0] for keyword_tup in similar_keywords
            ]

        keywords_dict[target] = list(
            {
                _process_keyword(keyword, lemmatize_model)
                for keyword in keywords_dict[target]
            }
        )

    return keywords_dict


if __name__ == "__main__":
    lemmatize_model = WordNetLemmatizer()

    # chose this model bc pretrained on wikipedia and news data - exactly like our dataset
    # learn more here: https://github.com/RaRe-Technologies/gensim-data
    word_similarity_model = gensim_api.load("fasttext-wiki-news-subwords-300")

    KEYWORDS_DICT = get_related_keywords(
        CHARMANDER_KEYWORDS_DICT, lemmatize_model, word_similarity_model
    )

    print(KEYWORDS_DICT)
