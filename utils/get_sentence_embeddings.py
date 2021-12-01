from copy import deepcopy
import gensim.downloader as gensim_api
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import string
import re
from keywords import CHARMELEON_KEYWORDS_DICT
import random
import mwparserfromhell
import shutil
import sys


def apply_and_concat(dataframe, field, func, column_names):
    return pd.concat(
        (
            dataframe,
            dataframe[field].apply(
                lambda cell: pd.Series(func(cell), index=column_names)
            ),
        ),
        axis=1,
    )


def _extract_relevant_sentences_by_row(sentences, keywords):
    """
    Extracts sentences from text that contain any of the keywords.

    Args:
        `keywords`: List of keywords
        `text`: Text to return sentences from

    Returns:
        List of sentences that contain at least one of the keywords.
    """
    relevant_sentences = {}

    for sentence in sentences:
        words = word_tokenize(sentence)
        count = 0
        for word in words:
            if word in keywords:
                count += 1

        if count not in relevant_sentences:
            relevant_sentences[count] = [sentence]
        else:
            relevant_sentences[count].append(sentence)

    max_count = max(relevant_sentences.keys())
    if max_count == 0:
        return None, None, None

    return (
        max_count,
        " ".join(relevant_sentences[max_count]),
        list({x for v in relevant_sentences.values() for x in v}),
    )


def _embed_sentences_by_row(x, sentence_embed_model):
    return np.mean(sentence_embed_model.encode(x), axis=0) if x else None


def write_metadata_sentences_embeddings(
    metadata, embeddings, relevant_sentences, country, target
):
    metadata_output_data_path = os.path.join(
        "data/intermediate_sentence_embeddings", country, target, "metadata.csv"
    )
    relevant_sentences_output_data_path = os.path.join(
        "data/relevant_sentences", country, target, "relevant_sentences.csv"
    )
    embeddings_output_data_path = os.path.join(
        "data/intermediate_sentence_embeddings", country, target, "embeddings.npy",
    )

    for dir in [
        os.path.join("data/intermediate_sentence_embeddings", country, target),
        os.path.join("data/relevant_sentences", country, target),
    ]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    with open(metadata_output_data_path, "a") as f:
        metadata.to_csv(f, header=f.tell() == 0, index=False)

    with open(relevant_sentences_output_data_path, "a") as f:
        relevant_sentences.to_csv(f, header=f.tell() == 0, index=False)

    if os.path.exists(embeddings_output_data_path):
        embeddings_old = np.load(embeddings_output_data_path)
        embeddings = np.concatenate([embeddings_old, embeddings])
    print(f"Embeddings shape: {embeddings.shape}")
    np.save(
        embeddings_output_data_path, embeddings,
    )


def embed_sentences(
    input_data_path, sentence_embed_model, target, keywords_dict,
):
    """
    Given metadata and articles stored in `input_data_path`,
    writes metadata and averaged sentence embeddings.

    Args:
        `input_data_path`: Input data path
        `target`: Target
        `sentence_embed_model`: Model for generating sentence embeddings
        `keywords_dict`: Model for keywords for each target

    Returns:
        None
    """
    df = pd.read_json(input_data_path)
    df["DHSID_EA"] = df["tag"].apply(lambda x: x[:-5])
    df["cname"] = df["DHSID_EA"].apply(lambda x: x[:2])

    # important because have amny articles that aren't associated with a real DHSID_EA
    df = df[df["cname"] != "no"]

    df = df[["DHSID_EA", "cname", "clean_text"]]

    print(f"Evaluating {df.shape[0]} examples")

    df[f"clean_sentences"] = df["clean_text"].apply(
        lambda x: mwparserfromhell.parse(x).strip_code().replace("\n", " ")
    )
    df[f"clean_sentences"] = df["clean_sentences"].apply(sent_tokenize)

    keywords = keywords_dict[target]
    print(f"Creating embeddings for target {target}")

    df = apply_and_concat(
        df,
        "clean_sentences",
        lambda x: _extract_relevant_sentences_by_row(x, keywords),
        ["relevance_score", "most_relevant_sentences", "all_relevant_sentences",],
    )

    df = df[~df["all_relevant_sentences"].isna()]

    print(f"Found relevant sentences for {df.shape[0]} examples")

    embeddings = np.zeros((df.shape[0], 384))
    for i, text in enumerate(df["all_relevant_sentences"]):
        embeddings[i, :] = _embed_sentences_by_row(text, sentence_embed_model)

    print(f"Writing embeddings for target {target} and countries {df.cname.unique()}")

    df = df.reset_index(drop=True)
    for country in df.cname.unique():
        country_idx = df[df.cname == country].index.values

        country_df = df.loc[
            country_idx, ["DHSID_EA", "relevance_score", "most_relevant_sentences",],
        ]

        write_metadata_sentences_embeddings(
            metadata=country_df[["DHSID_EA"]],
            relevant_sentences=country_df[
                ["DHSID_EA", "relevance_score", "most_relevant_sentences"]
            ],
            embeddings=embeddings[country_idx, :],
            country=country,
            target=target,
        )


if __name__ == "__main__":
    target = sys.argv[1]
    input_data_dir = "data/wikipedia"

    # good mixture of small size and high sentence embedding performance from sentence-transformers
    # learn more here: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    sentence_embed_model = SentenceTransformer("all-MiniLM-L12-v2")

    files_embedded = []
    for file in os.listdir(input_data_dir):
        print(f"Starting to embed {file}")
        embed_sentences(
            input_data_path=os.path.join(input_data_dir, file),
            sentence_embed_model=sentence_embed_model,
            target=target,
            keywords_dict=CHARMELEON_KEYWORDS_DICT,
        )
        files_embedded.append(file)
        print(f"Have embedded {files_embedded}\n")
