from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import DATA_DIR
from torch.utils.data import Dataset, DataLoader
from utils.get_data_loader import *

from utils.constants import (
    OUTPUT_SENTENCE_DATA_DIR,
    LABEL_METADATA_PATH,
    OUTPUT_DOCUMENT_DATA_DIR,
)


class SustainBenchTextTorchDataset(Dataset):
    def __init__(
            self,
            features,
            target,
            model_type,
            classification_threshold=0,
            data_dir=DATA_DIR,
            split_scheme="countries",
            data_split="train" # should be either train or test
    ):
        dataset = SustainBenchTextDataset(features=features, target=target, model_type=model_type, classification_threshold=classification_threshold, data_dir=data_dir, split_scheme=split_scheme)
        self.embeddings,self.labels = dataset.get_data(data_split)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class SustainBenchTextTorchTrainDatasetForNAS(Dataset):
    def __init__(self):
        dataset = SustainBenchTextDataset(features=["all_sentence"], target="asset_index", model_type="classification", classification_threshold=0, data_dir=DATA_DIR)
        self.embeddings,self.labels = dataset.get_data("train")

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SustainBenchTextTorchTestDatasetForNAS(Dataset):
    def __init__(self):
        dataset = SustainBenchTextDataset(features=["all_sentence"], target="asset_index", model_type="classification", classification_threshold=0, data_dir=DATA_DIR)
        self.embeddings,self.labels = dataset.get_data("test")

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]