import sys
import os
import math
sys.path.append('/Users/caravanuden/git-repos/Multimodal-deep-learning-for-poverty-prediction')

from utils.get_data_loader import SustainBenchTextDataset
from sklearn.linear_model import Ridge, LogisticRegression
import numpy as np
# from tune_sklearn import TuneGridSearchCV
import time # Just to compare fit times
from sklearn.metrics import r2_score, classification_report

from typing import List


from enum import Enum

from file_utils import *




class ModelType(Enum):
    regression = 1,
    classification = 2

class FeatureType(Enum):
    target_sentence=1,
    all_sentence=2,
    document=3,
    target_all_sentence=4,
    target_sentence_document=5,
    all_sentence_document=6



class ModelInterface:
    """
    This class serves as an interface for the models that we'll use in our experiments.
    Those models we use in experiments should ideally inherit from this interface
    """
    def __init__(self, name="No name provided"):
        self._name = name

    def name(self):
        return self._name

    def fit(self, X,y):
        pass

    def predict(self, X):
        pass





class Experiment:

    def __init__(self, classification_cutoff_dict: dict, targets: List[str], feature_types: List[FeatureType],
                 model_types: List[ModelType], models: List[object]):
        """

        :param classification_cutoff_dict:
        :param targets: what labels to predict on
        :param feature_types: List of features to use when loading dataset.
        All elements should be a value in ['target_sentence', 'all_sentence', 'document', 'target_all_sentence', 'target_sentence_document', 'all_sentence_document']:
        :param model_types: Whether models are classification or regression.
        This list should have the same size as the list of models!
        model_types[i] should be the type of model (Regression or Classification) corresponding to models[i]
        :param models: The list of model objects.
        These should all support the methods given in ModelInterface so that the Experiment class's functions
        will work for all models in the list.
        """
        assert len(model_types) == len(models), "model_types list must have same length as models list"
        self.classification_cutoff_dict = classification_cutoff_dict
        self.targets = targets
        self.feature_types = feature_types
        self.model_types = model_types
        self.models = models

    def get_model_name(self, model: object) -> str:
        """
        :param model: a model object (i.e., a model from the self.models array)
        :return: A string representing the name of the given model objec.t
        """
        return type(model).__name__


    def get_experiment_name(self, model: object, model_type: ModelType) -> str:
        """
        :param model:
        :param model_type:
        :return: A string representing the name of this experiment.
         (This is so that we can use this name for save files so that we'll know which results pertain to which
         experiment.)
        """
        feature_types_string = "-".join(self.feature_types)
        targets_string = "-".join(self.targets)
        return f"{self.get_model_name(model)}_{model_type.name}_{feature_types_string}_{targets_string}"

    def run_experiments(self):
        """
        Run all the experiments and print out the evaluation metrics in the relevant files.
        :return:
        """
        for model,model_type in zip(self.models, self.model_types):
            for target in self.targets:
                for feature_type in self.feature_types:
                    model = self.models[i]

                    ds = SustainBenchTextDataset(
                        data_dir=PATH_TO_DATA_DIR,
                        feature_type=feature_type,
                        target=target,
                        model_type=model_type,
                        classification_cutoff=self.classification_cutoff_dict[target]
                    )

                    train_X, train_y = ds.get_data('train')
                    test_X, test_y = ds.get_data('test')

                    start = time.time()
                    clf.fit(train_embeddings, train_labels)
                    end = time.time()
                    print(f"Fit Time for {self.get_model_name(i)} with target {target} and feature_type {feature_type}: {end - start}")









