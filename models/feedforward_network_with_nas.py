"""
Neural network with architecture search
NOTE: I got inspiration from https://github.com/jing-IG/nni-function-approximator/blob/master/func_approx_nni.py
"""

from typing import List
import torch
import nni.retiarii.nn.pytorch as nn
from experiments.experiment import ModelType, ModelInterface
from tqdm import trange
import numpy as np
import pdb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# pass in :
# [[values_for_hidden_dims_to_try], [values_for_hidden_dims_to_try], etc.
# in addition, we'll use nn.Identity() to select away from certain things sometimes; although, htat might not work
# (500, feature), (feature,whatever)->Identity



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeedforwardNetworkModuleForNAS(nn.Module):
    def __init__(self, input_dim, num_hidden_layers: int = 4, output_dim=1,
                 model_type: ModelType = ModelType.classification):
        super(FeedforwardNetworkModuleForNAS, self).__init__()

        NUM_HIDDEN_UNITS_LIST = [16,32,64,128,256,512]
        NUM_HIDDEN_UNITS_LIST_WITH_ZERO = [0,16,32,64,128,256,512]
        HIDDEN_ACTIVATIONS = [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()]

        self.hidden_dims_1 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST)
        self.hidden_dims_2 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)
        self.hidden_dims_3 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)
        self.hidden_dims_4 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)
        self.hidden_dims_5 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)

        self.fc1 = self.get_fc_block([input_dim, self.hidden_dims_1], HIDDEN_ACTIVATIONS)
        last_hidden_layer_dims = self.hidden_dims_1
        if num_hidden_layers >= 2:
            self.fc2 = self.get_fc_block([self.hidden_dims_1, self.hidden_dims_2], HIDDEN_ACTIVATIONS)
            last_hidden_layer_dims = self.hidden_dims_2
        if num_hidden_layers >= 3:
            self.fc3 = self.get_fc_block([self.hidden_dims_2, self.hidden_dims_3], HIDDEN_ACTIVATIONS)
            last_hidden_layer_dims = self.hidden_dims_3
        else:
            self.fc4 = self.get_fc_block([self.hidden_dims_3, self.hidden_dims_4], HIDDEN_ACTIVATIONS)
            last_hidden_layer_dims = self.hidden_dims_4

        if model_type == ModelType.regression:
            self.final_layer = nn.Sequential(
                nn.Linear(last_hidden_layer_dims, output_dim),
            )
        elif model_type == ModelType.classification:
            output_dim = 2 # we do this and use softmax below because BCE loss is not supported by nni pl.Classification
            self.final_layer = nn.Sequential(
                nn.Linear(last_hidden_layer_dims, output_dim),
                nn.Softmax()  # apply sigmoid to the output to get the probability
            )
        else:
            raise NotImplementedError("Only supports regression and classification")


    def get_fc_block(self, dims, activation_choices):
        return nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LayerChoice(activation_choices)
        )




    def forward(self, x):
        """

        :param x: the input to compute the forward pass on
        should be of shape (batch_size, input_shape)
        :return: the output. Could be an array or a number depending on the architecture definition.
        """
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.final_layer(output)

        return output