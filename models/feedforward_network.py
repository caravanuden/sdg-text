"""
This file implements a basic feedforward network which will take in inputs from
"""
from typing import List
import torch
import torch.nn as nn
from experiments.experiment import ModelType


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activations: List = None, model_type: ModelType = ModelType.regression):
        """

        :param input_dim: the dimensions of a single input vector
        :param hidden_dim: the number of hidden units
        :param output_dim: the number of outputs. In our case, as long as we're only doing binary classification
        and regression, this should always be 1. But, if we switch to multi-class, then it could be more than one.
        """
        super(FeedforwardNN, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = list()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

