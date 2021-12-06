from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
            feature_type,
            target,
            model_type,
            classification_cutoff=None,
            data_dir=OUTPUT_SENTENCE_DATA_DIR,
            split_scheme="countries",
            data_split="train" # should be either train or test
    ):
        dataset = SustainBenchTextDataset(feature_type, target, model_type, classification_cutoff, data_dir, split_scheme)
        self.embeddings,self.labels = dataset.get_data(data_split)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]