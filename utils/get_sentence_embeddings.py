import gensim.downloader as gensim_api
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


STARTER_KEYWORDS_DICT = {
    'asset_index': ['wealth', 'poverty'],
    'sanitation_index': ['sanitation'],
    'women_edu': ['woman', 'education'],
}


def get_related_keywords(starter_keywords_dict, n=100):
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
    # chose this model bc pretrained on wikipedia and news data - exactly like our dataset
    # learn more here: https://github.com/RaRe-Technologies/gensim-data
    model = gensim_api.load('fasttext-wiki-news-subwords-300')
    wordnet_lemmatizer = WordNetLemmatizer()

    for target, starter_keywords in starter_keywords_dict.items():
        starter_keywords_dict[target] += model.most_similar(starter_keywords, n=n)
        starter_keywords_dict[target] = [wordnet_lemmatizer.lemmatize(word) for word in starter_keywords_dict[target]]

    return starter_keywords_dict


def extract_relevant_sentences(keywords, article):
    """
    Extracts sentences from article that contain any of the keywords.

    Args:
        `keywords`: List of lemmatized keywords
        `article`: Article to return sentences from

    Returns:
        List of sentences that contain at least one of the keywords.
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    sentences = nltk.sent_tokenize(article)

    relevant_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentences)
        if any([wordnet_lemmatizer.lemmatize(word) in keywords for word in words]):
            relevant_sentences.append(sentence)

    return relevant_sentences


def embed_sentences(input_data_path, output_data_path, keywords_dict=None):
    """
    Given metadata and articles stored in `input_data_path`,
    writes metadata and averaged sentence embeddings to `output_data_path`.

    Args:
        `input_data_path`: Input data path
        `output_data_path`: Output data path

    Returns:
        None
    """
    if keywords_dict is None:
        keywords_dict = get_related_keywords(STARTER_KEYWORDS_DICT)

    # articles = read data from input_data_path, should return list of tuples (metadata, article)?

    # good mixture of small size and high sentence embedding performance from sentence-transformers
    # learn more here: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    model = SentenceTransformer('all-MiniLM-L12-v2')

    for target, keywords in keywords_dict.items():
        sentence_embeddings = []
        for metadata, article in articles:
            relevant_sentences = extract_relevant_sentences(keywords, article)
            avg_sentence_embedding = np.mean(model.encode(relevant_sentences), axis=0)
            sentence_embeddings.append(dict(metadata, **{'embedding': avg_sentence_embedding}))

        sentence_embeddings = pd.DataFrame(sentence_embeddings)
        sentence_embeddings.to_csv(os.path.join(output_data_path, target))
