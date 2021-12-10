"""
This file implements a basic feedforward network which will take in inputs from.
Note: we can't set the input dimension until we call the fit function (because the dimsensionality of different
inputs differs; for some it's 300 dimensional, for others 384 dimensional.)
"""
from typing import List
import torch
import torch.nn as nn
from experiments.experiment import ModelType, ModelInterface
from tqdm import trange
import numpy as np
import pdb
import torch.nn.functional as F
from torch.autograd import Variable


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def convert_to_one_hot(array, size=2):
    """
    Note: Got help from https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    """
    new_arr = np.zeros((array.size, size))
    new_arr[np.arange(array.size), array] = 1
    return new_arr


class FeedforwardNetworkModule(nn.Module):
    def __init__(self, input_dim, hidden_dims: List[int], output_dim=1, activations: List[object]=list(),
                 model_type: ModelType = ModelType.regression, default_hidden_activation=nn.Sigmoid()):
        """
        :param input_dim: the input data dimension
        :param hidden_dims: list of hidden units for each hidden dimension
        :param output_dim
        :param activatiosn: the activations to be used in each hidden layer of the network; activations[i] is the
        activation to be used at layer i
        :param default_hidden_activation: this is used only if activations list is not specified.
        In that case, we just use the default activation at each layer.
        :param output_dim: the number of outputs. In our case, as long as we're only doing binary classification
        and regression, this should always be 1. But, if we switch to multi-class, then it could be more than one.


        NOTE: the input dimension must be inferred when fit() is called.
        """
        super(FeedforwardNetworkModule, self).__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activations = activations
        self.model_type = model_type
        self.default_hidden_activation = default_hidden_activation

        # Configure the architecture
        self.configure_architecture(input_dim)

        if self.model_type == ModelType.classification:
            self.output_dim = 2 # we'll use cross entropy loss now

    def configure_architecture(self, input_dim):
        self.layers = list()

        # set up layer dims
        layer_dims = [input_dim]
        layer_dims.extend(self.hidden_dims)
        layer_dims.append(self.output_dim)

        # determine if we need to build up activations list (or if it was specified in model constructor)
        build_activations = False
        if len(self.activations) == 0:
            build_activations = True

        # create all layers and activations
        for i in range(1, len(layer_dims)):
            self.layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            if build_activations and i != len(layer_dims) - 1:
                self.activations.append(self.default_hidden_activation)

        # the last activation is specified by the task at hand!
        final_activation = None
        if self.model_type == ModelType.classification:
            #final_activation = nn.Sigmoid()
            final_activation = None
        elif self.model_type == ModelType.regression:
            final_activation = None  # keep it at None for regression model
        else:
            raise NotImplementedError
        self.activations.append(final_activation)

        # make sure that torch nn module registers the layers we added
        # so that it'll automatically populate model.modules() and model.parameters()
        self.layers = nn.ModuleList(self.layers)






    def forward(self, x):
        """

        :param x: the input to compute the forward pass on
        should be of shape (batch_size, input_shape)
        :return: the output. Could be an array or a number depending on the architecture definition.
        """
        output = x
        for layer,activation in zip(self.layers, self.activations):
            output = layer(output)
            if activation is not None:
                output = activation(output)
        return output


class FeedforwardNewtork(ModelInterface):
    def __init__(self, hidden_dims: List[int], output_dim=1, activations: List[object] = list(),
                 model_type: ModelType = ModelType.regression, default_hidden_activation: object = nn.Sigmoid(),
                 batch_size: int = 32, num_epochs: int = 10, optimizer: object = torch.optim.Adam,
                 criterion: object = None, learning_rate: float = 0.0001):
        """
        For description of parameters not listed here, see FeedforwardNetworkModule

        :param batch_size: the batch size for trainig
        :param num_epochs: the number of training epochs to use when fitting the model to the data.
        :param optimizer: optimizer to use for training
        :param criterion: the type of loss to use
        :param learning_rate: the learning rate to use for training.
        """
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim


        # self.hidden_layer_activation = activation
        self.activations = activations
        self.model_type = model_type
        self.default_hidden_activation = default_hidden_activation
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

        if self.model_type == ModelType.classification:
            self.output_dim = 2  # we'll use cross entropy loss now

        # these will be configured in fit()
        # the model needs the input dims to be initialized correctly
        self.model = None
        self.optimizer = None


    def reset(self, input_dim: int):
        """

        :param input_dim: the dimensions of the inputs
        :return:
        """
        # first, re-configure the architecture.
        #pdb.set_trace()
        self.model = FeedforwardNetworkModule(input_dim, self.hidden_dims, self.output_dim, self.activations,
                                              self.model_type, self.default_hidden_activation).to(DEVICE)
        #self.model.configure_architecture(input_dims)
        #self.model = self.model.to(DEVICE)
        self.optimizer = self.user_specified_optimizer(self.model.parameters(), self.learning_rate)

    def fit(self,x,y):
        """

        :param x: the inputs of shape (num_inputs, input_shape)
        :param y: their corresponding labels of shape (num_inputs,)
        :return: nothing. just fit the model
        """
        # first, re-configure the architecture.
        self.reset(x.shape[-1])

        for j in range(self.num_epochs):
            self.optimizer.zero_grad()  # zero the gradient buffers

            # now, fit the model
            mini_batch_range = trange(x.shape[0] // self.batch_size + 1)

            for i in mini_batch_range:
                end_range = min((i + 1) * self.batch_size, x.shape[0])

                # get batches
                batch_X = x[i*self.batch_size : end_range]
                batch_y = y[i*self.batch_size : end_range]
                batch_X = torch.from_numpy(batch_X).float()
                batch_y = torch.from_numpy(convert_to_one_hot(batch_y.astype(int))).float()

                # get the loss and do backprop
                outputs = self.model(batch_X)
                #pdb.set_trace()
                loss = self.criterion(outputs, Variable(batch_y))
                print(loss)
                loss.backward()

                # output tqdm thing
                mini_batch_range.set_postfix(loss=loss.item())

    def predict_proba(self, test_x):
        """

        :param test_x: the points to predict on.
        should have shape (train_set_size, input_dims)
        :return: the predictions.
        Will be an array of shape (train_set_size, output_dims), where output_dims will not be one only for
        multi-class classification.
        """
        outputs = np.zeros((test_x.shape[0], self.output_dim))

        with torch.no_grad():
            mini_batch_range = trange(test_x.shape[0] // self.batch_size + 1)
            for i in mini_batch_range:
                end_range = min((i + 1) * self.batch_size, test_x.shape[0])

                # get batches
                batch_X = test_x[i * self.batch_size: end_range]
                batch_X = torch.from_numpy(batch_X).float()

                predictions = self.model(batch_X)
                predictions = F.softmax(predictions)
                outputs[i * self.batch_size: end_range] = predictions.cpu().numpy()

        return outputs


    def predict(self, test_x):
        """

        :param test_x: the points to predict on.
        should have shape (train_set_size, input_dims)
        :return: the predictions.
        Will be an array of shape (train_set_size,) with the classification deciosn
        """
        output_probs = self.predict_proba(test_x)

        preds = np.argmax(output_probs, axis=1)

        #probs = np.copy(output_probs)
        #probs[probs<0.5] = 0
        #probs[probs>=0.5] = 1
        return preds
