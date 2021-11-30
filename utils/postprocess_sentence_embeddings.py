from copy import deepcopy
import gensim.downloader as gensim_api
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import string
import re
from keywords import CHARMELEON_KEYWORDS_DICT
import random
import mwparserfromhell


def _extract_relevant_sentences_by_row(text, keywords, lemmatize_model=None):
    """
    Extracts sentences from text that contain any of the keywords.

    Args:
        `keywords`: List of lemmatized keywords
        `text`: Text to return sentences from

    Returns:
        List of sentences that contain at least one of the keywords.
    """
    if not text:
        return None

    parsed_wikicode = mwparserfromhell.parse(text)
    text = parsed_wikicode.strip_code()

    sentences = text.split(".")
    relevant_sentences = []

    for sentence in sentences:
        words = sentence.split(" ")
        if any(
            [
                lemmatize_model.lemmatize(word) if lemmatize_model else word in keywords
                for word in words
            ]
        ):
            relevant_sentences.append(sentence)

    return relevant_sentences if relevant_sentences else None


def _embed_sentences_by_row(x, sentence_embed_model):
    return np.mean(sentence_embed_model.encode(x), axis=0) if x else None


def write_metadata_and_embeddings(df, metadata_cols, embedding_cols, output_data_dir):
    metadata_output_data_path = os.path.join(output_data_dir, "metadata.csv")
    embeddings_output_data_path = os.path.join(output_data_dir, "embeddings.npy")

    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)

    with open(metadata_output_data_path, "a") as f:
        df[metadata_cols].to_csv(f, header=f.tell() == 0, index=False)
    np.save(
        embeddings_output_data_path, df[embedding_cols].to_numpy(),
    )


def embed_sentences(
    input_data_path,
    output_data_dir,
    sentence_embed_model,
    keywords_dict,
    lemmatize_model=None,
):
    """
    Given metadata and articles stored in `input_data_path`,
    writes metadata and averaged sentence embeddings to `output_data_dir`.

    Args:
        `input_data_path`: Input data path
        `output_data_dir`: Output data dir
        `lemmatize_model`: Model for lemmatizing words
        `sentence_embed_model`: Model for generating sentence embeddings
        `keywords_dict`: Model for keywords for each target

    Returns:
        None
    """
    df = pd.read_json(input_data_path)
    df["DHSID_EA"] = df["tag"].apply(lambda x: x[:-5])
    df["cname"] = df["DHSID_EA"].apply(lambda x: x[:2])
    df = df[df["cname"] != "no"]
    df = df[["DHSID_EA", "cname", "clean_text"]]

    print(f"Evaluating {df.shape[0]} examples")

    for target, keywords in keywords_dict.items():
        print(f"Creating embeddings for target {target}")

        df[f"clean_text_{target}"] = df["clean_text"].apply(
            lambda x: _extract_relevant_sentences_by_row(x, keywords, lemmatize_model)
        )
        df_for_target = df[~df[f"clean_text_{target}"].isna()].reset_index(drop=True)

        print(f"Found relevant sentences for {df_for_target.shape[0]} examples")

        df_for_target["embedding"] = df_for_target[f"clean_text_{target}"].apply(
            lambda x: _embed_sentences_by_row(x, sentence_embed_model)
        )

        for country in df_for_target.cname.unique():
            print(f"Writing embeddings for target {target} and country {country}")
            write_metadata_and_embeddings(
                df=df_for_target[df_for_target.cname == country],
                metadata_cols=["DHSID_EA"],
                embedding_cols=["embedding"],
                output_data_dir=os.path.join(
                    output_data_dir, country, f"{target}_sentence_embedding"
                ),
            )


if __name__ == "__main__":
    data_dir = "data"
    input_data_dir = os.path.join(data_dir, "intermediate_embeddings")
    output_data_dir = os.path.join(data_dir, "embeddings")

    for country in countries:
        for target in targets:
            ids = pd.read_csv(
                os.path.join(input_data_dir, country, target, "metadata.csv")
            )
            metadata = pd.read_csv(os.path.join(data_dir, "dhs_final_labels.csv"))
            metadata = metadata.merge(ids, on="DHSID_EA")[
                ["DHSID_EA", "lat", "lon", target]
            ]
            locations = metadata.groupby(["lat", "lon"])["DHSID_EA"].apply(list)

            for location in locations:
