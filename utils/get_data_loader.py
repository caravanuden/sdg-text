import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from utils.constants import (
    OUTPUT_SENTENCE_DATA_DIR,
    LABEL_METADATA_PATH,
    OUTPUT_DOCUMENT_DATA_DIR,
)

SPLITS = {
    "train": [
        "AL",
        "BD",
        "CD",
        "CM",
        "GH",
        "GU",
        "HN",
        "IA",
        "ID",
        "JO",
        "KE",
        "KM",
        "LB",
        "LS",
        "MA",
        "MB",
        "MD",
        "MM",
        "MW",
        "MZ",
        "NG",
        "NI",
        "PE",
        "PH",
        "SN",
        "TG",
        "TJ",
        "UG",
        "ZM",
        "ZW",
        "BF",
        "BJ",
        "BO",
        "CO",
        "DR",
        "GA",
        "GN",
        "GY",
        "HT",
        "NM",
        "SL",
        "TD",
        "TZ",
    ],
    "test": [
        "AM",
        "AO",
        "BU",
        "CI",
        "EG",
        "ET",
        "KH",
        "KY",
        "ML",
        "NP",
        "PK",
        "RW",
        "SZ",
    ],
}


def split_by_countries(idxs, ood_countries, metadata):
    countries = np.asarray(metadata["country"].iloc[idxs])
    is_ood = np.any([(countries == country) for country in ood_countries], axis=0)
    return idxs[~is_ood], idxs[is_ood]


class SustainBenchTextDataset:
    def __init__(
        self,
        feature_type,
        target,
        model_type,
        classification_cutoff=None,
        data_dir=OUTPUT_SENTENCE_DATA_DIR,
        split_scheme="countries",
    ):
        self.data_dir = data_dir
        # feature_type in ['target_sentence', 'all_sentence', 'target_all_sentence', 'target_sentence_doc', 'all_sentence_doc',]
        self.feature_type = feature_type

        # feature_type in ['asset_index', 'sanitation_index', 'water_index', 'women_education',]
        self.target = target

        # model_type in ['classification', 'regression']
        self.model_type = model_type

        self.classification_cutoff = classification_cutoff
        self.split_scheme = split_scheme

    def get_data(self, split):
        if split not in ["train", "test"]:
            raise ValueError(f"split {split} must be one of ['train', 'test']")
        embeddings = []
        labels = []
        for country in SPLITS[split]:
            if self.feature_type == "target_sentence":
                country_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        self.target,
                        "metadata.csv",
                    )
                )
                country_embeddings = np.load(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        self.target,
                        "embeddings.npy",
                    )
                )
            if self.feature_type == "all_sentence":
                country_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        "all",
                        "metadata.csv",
                    )
                )
                country_embeddings = np.load(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        "all",
                        "embeddings.npy",
                    )
                )
            if self.feature_type == "document":
                country_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir, "document_embeddings", country, "metadata.csv"
                    )
                )
                country_embeddings = np.load(
                    os.path.join(
                        self.data_dir, "document_embeddings", country, "embeddings.npy"
                    )
                )
            if self.feature_type == "target_all_sentence":
                country_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        self.target,
                        "metadata.csv",
                    )
                )
                country_all_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        "all",
                        "metadata.csv",
                    )
                )
                country_target_embeddings = np.load(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        self.target,
                        "embeddings.npy",
                    )
                )
                country_all_embeddings = np.load(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        "all",
                        "embeddings.npy",
                    )
                )

                country_embeddings = np.zeros(
                    (country_metadata.shape[0], country_target_embeddings.shape[1] * 2,)
                )
                for i, location in enumerate(country_metadata.DHSID_EA):
                    target_embedding_for_loc = country_target_embeddings[
                        country_metadata.DHSID_EA == location, :
                    ]
                    all_embedding_for_loc = country_all_embeddings[
                        country_all_metadata.DHSID_EA == location, :
                    ]

                    country_embeddings[i, :] = np.concatenate(
                        [target_embedding_for_loc, all_embedding_for_loc], axis=1
                    )
            if self.feature_type == "target_sentence_document":
                country_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        self.target,
                        "metadata.csv",
                    )
                )
                country_target_embeddings = np.load(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        self.target,
                        "embeddings.npy",
                    )
                )

                country_document_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir, "document_embeddings", country, "metadata.csv"
                    )
                )
                country_document_embeddings = np.load(
                    os.path.join(
                        self.data_dir, "document_embeddings", country, "embeddings.npy"
                    )
                )

                country_embeddings = np.zeros(
                    (
                        country_metadata.shape[0],
                        country_target_embeddings.shape[1]
                        + country_document_embeddings.shape[1],
                    )
                )
                for i, location in enumerate(country_metadata.DHSID_EA):
                    target_embedding_for_loc = country_target_embeddings[
                        country_metadata.DHSID_EA == location, :
                    ]
                    document_embedding_for_loc = np.mean(
                        country_document_embeddings[
                            country_document_metadata.DHSID_EA == location, :
                        ],
                        axis=0,
                    )
                    document_embedding_for_loc = np.expand_dims(
                        document_embedding_for_loc, axis=0
                    )

                    country_embeddings[i, :] = np.concatenate(
                        [target_embedding_for_loc, document_embedding_for_loc], axis=1
                    )

            if self.feature_type == "all_sentence_document":
                country_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        "all",
                        "metadata.csv",
                    )
                )
                country_all_embeddings = np.load(
                    os.path.join(
                        self.data_dir,
                        "sentence_embeddings",
                        country,
                        "all",
                        "embeddings.npy",
                    )
                )

                country_document_metadata = pd.read_csv(
                    os.path.join(
                        self.data_dir, "document_embeddings", country, "metadata.csv"
                    )
                )
                country_document_embeddings = np.load(
                    os.path.join(
                        self.data_dir, "document_embeddings", country, "embeddings.npy"
                    )
                )

                country_embeddings = np.zeros(
                    (
                        country_metadata.shape[0],
                        country_all_embeddings.shape[1]
                        + country_document_embeddings.shape[1],
                    )
                )
                for i, location in enumerate(country_metadata.DHSID_EA):
                    all_embedding_for_loc = country_all_embeddings[
                        country_metadata.DHSID_EA == location, :
                    ]
                    document_embedding_for_loc = np.mean(
                        country_document_embeddings[
                            country_document_metadata.DHSID_EA == location, :
                        ],
                        axis=0,
                    )
                    document_embedding_for_loc = np.expand_dims(
                        document_embedding_for_loc, axis=0
                    )

                    country_embeddings[i, :] = np.concatenate(
                        [all_embedding_for_loc, document_embedding_for_loc], axis=1
                    )

            labels += list(country_metadata[self.target])
            embeddings.append(country_embeddings)

        labels = np.array(labels)
        embeddings = np.concatenate(embeddings, axis=0)

        non_nan_idxs = [~np.isnan(label) for label in labels]
        labels = labels[non_nan_idxs]
        if self.model_type == "classification":
            labels = labels < self.classification_cutoff
            labels = labels.astype(int)
        embeddings = embeddings[non_nan_idxs, :]

        return embeddings, labels
