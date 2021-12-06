import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.decomposition import PCA

from utils.constants import (
    OUTPUT_SENTENCE_DATA_DIR,
    LABEL_METADATA_PATH,
    OUTPUT_DOCUMENT_DATA_DIR,
    SENTENCE_EMBEDDING_SIZE,
    DOCUMENT_EMBEDDING_SIZE,
    PCA_SENTENCE_EMBEDDING_SIZE,
)

EMBEDDING_SIZES = {
    "target_sentence": SENTENCE_EMBEDDING_SIZE,
    "all_sentence": PCA_SENTENCE_EMBEDDING_SIZE,
    "document": DOCUMENT_EMBEDDING_SIZE,
}

CLASS_REBALANCE_RATIO = 0.3
CLASS_REBALANCE_RATIO_FOR_IMBLEARN = CLASS_REBALANCE_RATIO / (1 - CLASS_REBALANCE_RATIO)

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
        "NP",
        "PK",
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
        "ET",
        "EG",
        "KH",
        "KY",
        "ML",
        "PE",
        "PH",
        "RW",
        "SZ",
    ],
}


class SustainBenchTextDataset:
    def __init__(
        self,
        features,
        target,
        model_type,
        key="DHSID_EA",
        data_dir=OUTPUT_SENTENCE_DATA_DIR,
        split_scheme="countries",
        classification_threshold=None,
        rebalance=False,
    ):
        self.key = key
        self.data_dir = data_dir
        # each feature in ['target_sentence', 'all_sentence', 'document'], we'll concat these feature together
        self.features = features

        # target in ['asset_index', 'sanitation_index', 'water_index', 'women_education',]
        self.target = target

        # model_type in ['classification', 'regression']
        self.model_type = model_type

        self.classification_threshold = classification_threshold
        self.split_scheme = split_scheme
        self.rebalance = rebalance
        self.pca_model = None

    def _get_metadata_and_embeddings_for_country(self, country, feature, target):
        data_dir = (
            os.path.join(self.data_dir, f"{feature}_embeddings", country)
            if feature == "document"
            else os.path.join(self.data_dir, f"{feature}_embeddings", country, target)
        )
        metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv",))
        embeddings = np.load(os.path.join(data_dir, "embeddings.npy",))

        return metadata, embeddings

    def _train_pca_model(self):
        embeddings = []
        for country in SPLITS["train"]:
            _, country_embeddings = self._get_metadata_and_embeddings_for_country(
                country=country, feature="sentence", target="concatenated"
            )
            embeddings.append(country_embeddings)
        embeddings = np.concatenate(embeddings, axis=0)
        pca = PCA(n_components=PCA_SENTENCE_EMBEDDING_SIZE)
        self.pca_model = pca.fit(embeddings)
        print(
            f"The PCA model explains {round(np.sum(self.pca_model.explained_variance_) * 100, 3)}% of the variance in the embedding data"
        )

    def get_data(self, split):
        if split not in ["train", "test"]:
            raise ValueError(f"split {split} must be one of ['train', 'test']")

        labels = []
        embeddings = []

        for country in SPLITS[split]:
            metadata_list_by_feature = []
            embeddings_list_by_feature = []
            for feature in self.features:
                if feature == "target_sentence":
                    (
                        country_metadata,
                        country_embeddings,
                    ) = self._get_metadata_and_embeddings_for_country(
                        country=country, feature="sentence", target=self.target
                    )

                if feature == "all_sentence":
                    (
                        country_metadata,
                        country_embeddings,
                    ) = self._get_metadata_and_embeddings_for_country(
                        country=country, feature="sentence", target="concatenated"
                    )

                    if not self.pca_model:
                        self._train_pca_model()
                    country_embeddings = self.pca_model.transform(country_embeddings)

                if feature == "document":
                    (
                        country_metadata,
                        country_embeddings,
                    ) = self._get_metadata_and_embeddings_for_country(
                        country=country, feature="document", target=self.target
                    )

                metadata_list_by_feature.append(country_metadata)
                embeddings_list_by_feature.append(country_embeddings)

            all_country_metadata = pd.concat(metadata_list_by_feature)
            country_metadata = all_country_metadata.drop_duplicates(subset=[self.key])

            locations = country_metadata[self.key]
            country_embeddings = np.zeros(
                (
                    len(locations),
                    sum([EMBEDDING_SIZES[feature] for feature in self.features]),
                )
            )
            good_idxs = np.full((len(locations)), True)
            for i, location in enumerate(locations):
                embeddings_for_loc = []
                for j, feature_embedding in enumerate(embeddings_list_by_feature):
                    feature_embedding_for_loc = feature_embedding[
                        metadata_list_by_feature[j][self.key] == location, :
                    ]
                    if feature_embedding_for_loc.shape[0] == 0:
                        # will drop these later, just doing this so concat doesn't break
                        feature_embedding_for_loc = np.zeros(
                            (feature_embedding.shape[1])
                        )
                        feature_embedding_for_loc = np.expand_dims(
                            feature_embedding_for_loc, 0
                        )
                        good_idxs[i] = False
                    embeddings_for_loc.append(feature_embedding_for_loc)
                country_embeddings[i, :] = np.concatenate(embeddings_for_loc, axis=1)

            metadata_and_labels = pd.read_csv(
                os.path.join(self.data_dir, "dhs_final_labels.csv")
            )
            country_metadata_and_labels = country_metadata.merge(
                metadata_and_labels, on=self.key, how="left"
            )[[self.key, self.target]]

            country_labels = np.array(country_metadata_and_labels[self.target])
            country_labels = country_labels[np.where(good_idxs)]
            country_embeddings = country_embeddings[np.where(good_idxs), :]

            labels.append(country_labels)
            embeddings.append(country_embeddings)

        labels = np.concatenate(labels, axis=0)
        embeddings = np.concatenate(embeddings, axis=1)
        embeddings = np.squeeze(embeddings, axis=0)

        non_nan_idxs = [~np.isnan(label) for label in labels]
        labels = labels[non_nan_idxs]
        embeddings = embeddings[non_nan_idxs, :]

        if self.model_type == "classification":
            labels = labels < self.classification_threshold
            labels = labels.astype(int)

            if self.rebalance and split == "train":
                class_ratio = sum(labels) / len(labels)

                # get minority class ratio
                class_ratio = 1 - class_ratio if class_ratio > 0.5 else class_ratio
                class_ratio_for_imblearn = class_ratio / (1 - class_ratio)

                if class_ratio_for_imblearn < CLASS_REBALANCE_RATIO_FOR_IMBLEARN:
                    smote_tomek = SMOTETomek(
                        sampling_strategy=CLASS_REBALANCE_RATIO_FOR_IMBLEARN,
                        tomek=TomekLinks(sampling_strategy="majority"),
                    )
                    embeddings, labels = smote_tomek.fit_resample(embeddings, labels)
                else:
                    print(
                        f"No need to resample, classes are balanced within our tolerated class ratio of {CLASS_REBALANCE_RATIO}"
                    )

        return embeddings, labels
