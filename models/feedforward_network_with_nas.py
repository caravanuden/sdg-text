"""
Neural network with architecture search
NOTE: I got inspiration from https://github.com/jing-IG/nni-function-approximator/blob/master/func_approx_nni.py
"""

from typing import List
import torch
import nni.retiarii.nn.pytorch as nn
from experiments.experiment import *
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

        self.num_hidden_layers = num_hidden_layers

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
        if num_hidden_layers >= 4:
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
        if self.num_hidden_layers >= 2:
            output = self.fc2(output)
        if self.num_hidden_layers >= 3:
            output = self.fc3(output)
        if self.num_hidden_layers >= 4:
            output = self.fc4(output)
        output = self.final_layer(output)

        return output



class FeedforwardNetworkForNASModelInterface(ModelInterface):
    def __init__(self, model_module, name="NAS_selected_model", model_type: ModelType = ModelType.regression,
                 batch_size: int = 32, num_epochs: int = 10, optimizer: object = torch.optim.SGD,
                 criterion: object = None, learning_rate: float = 0.001):


        self.name = name

        self.model_type = model_type

        self.user_specified_optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        if criterion is None:
            if model_type == ModelType.classification:
                self.criterion = torch.nn.CrossEntropyLoss()
            elif model_type == ModelType.regression:
                self.criterion = torch.nn.MSELoss()
            else:
                raise NotImplementedError

        if model_type == ModelType.classification:
            self.output_dim=2
        else:
            self.output_dim=1

        # these will be configured in fit()
        # the model needs the input dims to be initialized correctly
        self.model_module = model_module
        self.model = model_module()
        self.optimizer = None

    def reset(self):
        self.model = self.model_module()
        self.optimizer = self.user_specified_optimizer(self.model.parameters(), self.learning_rate)

    def fit(self, x, y):
        """

        :param x: the inputs of shape (num_inputs, input_shape)
        :param y: their corresponding labels of shape (num_inputs,)
        :return: nothing. just fit the model
        """
        # first, re-configure the architecture.
        self.reset()

        # now, fit the model
        mini_batch_range = trange(x.shape[0] // self.batch_size + 1)
        for i in mini_batch_range:
            end_range = min((i + 1) * self.batch_size, x.shape[0])

            # get batches
            batch_X = x[i * self.batch_size: end_range]
            batch_y = y[i * self.batch_size: end_range]
            batch_X = torch.from_numpy(batch_X).float()
            batch_y = torch.from_numpy(batch_y).long()
            #batch_y = batch_y.reshape()

            # get the loss and do backprop
            self.optimizer.zero_grad()  # zero the gradient buffers
            outputs = self.model(batch_X)
            #pdb.set_trace()
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            # output tqdm thing
            mini_batch_range.set_postfix(loss=loss.item())

    def predict(self, test_x):
        """

        :param test_x: the points to predict on.
        should have shape (train_set_size, input_dims)
        :return: the predictions.
        """
        outputs = np.zeros((test_x.shape[0], 1))

        with torch.no_grad():
            mini_batch_range = trange(test_x.shape[0] // self.batch_size + 1)
            for i in mini_batch_range:
                end_range = min((i + 1) * self.batch_size, test_x.shape[0])

                # get batches
                batch_X = test_x[i * self.batch_size: end_range]
                batch_X = torch.from_numpy(batch_X).float()

                predictions = self.model(batch_X).cpu().numpy()
                if self.model_type == ModelType.classification:
                    predictions = np.argmax(predictions, axis=1).reshape(-1,1)
                outputs[i * self.batch_size: end_range] = predictions
        return outputs
