import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from utils.constants import OUTPUT_DATA_DIR

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
    ],
    "val": [
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
        self, feature_type, target, data_dir=OUTPUT_DATA_DIR, split_scheme="countries"
    ):
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.target = target
        self.split_scheme = split_scheme

    def get_data(self, split):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split {split} must be one of ['train', 'val', 'test']")
        embeddings = []
        labels = []
        for country in SPLITS[split]:
            print(labels, embeddings)
            country_metadata = pd.read_csv(
                os.path.join(OUTPUT_DATA_DIR, country, self.target, "metadata.csv")
            )
            labels += list(country_metadata[self.target])

            country_embeddings = np.load(
                os.path.join(OUTPUT_DATA_DIR, country, self.target, "embeddings.npy")
            )
            embeddings.append(country_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings, labels
