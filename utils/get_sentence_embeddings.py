from copy import deepcopy
import gensim.downloader as gensim_api
from nltk.stem import WordNetLemmatizer
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


def _extract_relevant_sentences_by_row(sentences, keywords, lemmatize_model=None):
    """
    Extracts sentences from text that contain any of the keywords.

    Args:
        `keywords`: List of lemmatized keywords
        `text`: Text to return sentences from

    Returns:
        List of sentences that contain at least one of the keywords.
    """
    relevant_sentences = {}

    for sentence in sentences:
        words = word_tokenize(sentence)
        count = 0
        for word in words:
            word = lemmatize_model.lemmatize(word) if lemmatize_model else word
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
    metadata, clean_sentences, embeddings, relevant_sentences, country, target
):
    metadata_output_data_path = os.path.join(
        "data/intermediate_sentence_embeddings", country, target, "metadata.csv"
    )
    clean_sentences_output_data_path = os.path.join(
        "data/clean_sentences", country, target, "clean_sentences.csv"
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
        os.path.join("data/clean_sentences", country, target),
    ]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    with open(metadata_output_data_path, "a") as f:
        metadata.to_csv(f, header=f.tell() == 0, index=False)

    with open(clean_sentences_output_data_path, "a") as f:
        clean_sentences.to_csv(f, header=f.tell() == 0, index=False)

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

    df[f"clean_sentences"] = df["clean_text"].apply(
        lambda x: mwparserfromhell.parse(x).strip_code().replace("\n", " ")
    )
    df[f"clean_sentences"] = df["clean_sentences"].apply(sent_tokenize)

    for target, keywords in keywords_dict.items():
        print(f"Creating embeddings for target {target}")

        df = apply_and_concat(
            df,
            "clean_sentences",
            lambda x: _extract_relevant_sentences_by_row(x, keywords, lemmatize_model),
            [
                f"relevance_score_{target}",
                f"most_relevant_sentences_{target}",
                f"all_relevant_sentences_{target}",
            ],
        )

        df_for_target = df[~df[f"all_relevant_sentences_{target}"].isna()]

        print(f"Found relevant sentences for {df_for_target.shape[0]} examples")

        embeddings = np.zeros((df_for_target.shape[0], 384))
        for i, text in enumerate(df_for_target[f"all_relevant_sentences_{target}"]):
            embeddings[i, :] = _embed_sentences_by_row(text, sentence_embed_model)

        print(
            f"Writing embeddings for target {target} and countries {df_for_target.cname.unique()}"
        )

        df_for_target = df_for_target.reset_index(drop=True)
        for country in df_for_target.cname.unique():
            country_idx = df_for_target[df_for_target.cname == country].index.values

            curr_df = df_for_target.loc[
                country_idx,
                [
                    "DHSID_EA",
                    f"relevance_score_{target}",
                    "clean_sentences",
                    f"most_relevant_sentences_{target}",
                ],
            ]
            curr_df.columns = [
                "DHSID_EA",
                "relevance_score",
                "clean_sentences",
                "most_relevant_sentences",
            ]

            write_metadata_sentences_embeddings(
                metadata=curr_df[["DHSID_EA"]],
                clean_sentences=curr_df[["DHSID_EA", "clean_sentences"]],
                relevant_sentences=curr_df[
                    ["DHSID_EA", "relevance_score", "most_relevant_sentences"]
                ],
                embeddings=embeddings[country_idx, :],
                country=country,
                target=target,
            )


if __name__ == "__main__":
    input_data_dir = "data/wikipedia"
    output_data_dir = "data/intermediate_sentence_embeddings"

    lemmatize_model = WordNetLemmatizer()

    # good mixture of small size and high sentence embedding performance from sentence-transformers
    # learn more here: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    sentence_embed_model = SentenceTransformer("all-MiniLM-L12-v2")

    files_embedded = []
    for file in os.listdir(input_data_dir):
        print(f"Starting to embed {file}")
        embed_sentences(
            input_data_path=os.path.join(input_data_dir, file),
            output_data_dir=output_data_dir,
            sentence_embed_model=sentence_embed_model,
            keywords_dict=CHARMELEON_KEYWORDS_DICT,
            # lemmatize_model=lemmatize_model,
        )
        files_embedded.append(file)
        print(f"Have embedded {files_embedded}\n")
