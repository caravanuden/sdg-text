from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

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
    def __init__(self, data_dir, feature_type, target_name, split_scheme="countries"):
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.target_name = target_name
        self.split_scheme = split_scheme

    def get_train_data():
        train_metadata
