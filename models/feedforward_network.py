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


class FeedforwardNetworkModule(nn.Module):
    def __init__(self, hidden_dims: List[int], output_dim=1, activations: List[object]=list(),
                 model_type: ModelType = ModelType.regression, default_hidden_activation=nn.Sigmoid(),
                 num_epochs=10):
        """
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
        #self.hidden_layer_activation = activation
        self.activations = activations
        self.model_type = model_type
        self.default_hidden_activation = default_hidden_activation

        # self.layers is what we'll use when we call forward.
        self.layers = list()

    def configure_architecture(self, input_dim):
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
            self.layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            if build_activations and i != len(layer_dims)-1:
                self.activations.append(self.default_hidden_activation)

        # the last activation is specified by the task at hand!
        final_activation = None
        if self.model_type == ModelType.classification:
            final_activation = nn.Sigmoid()
        elif self.model_type == ModelType.regression:
            final_activation = None # keep it at None for regression model
        else:
            raise NotImplementedError
        self.activations.append(final_activation)


    def forward(self, x):
        """

        :param x: the input to compute the forward pass on
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
                 batch_size: int = 32, num_epochs: int = 10, optimizer: object = torch.optim.SGD,
                 criterion: object = None, learning_rate: float = 0.05):
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

        self.model = FeedforwardNetworkModule(self.hidden_dims, self.output_dim, self.activations,
                                              self.model_type, self.default_hidden_activation)
        self.optimizer = optimizer(self.model.parameters(), learning_rate)
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




    def fit(self,x,y):
        """

        :param x: the inputs of shape (num_inputs, input_shape)
        :param y: their corresponding labels of shape (num_inputs,)
        :return: nothing. just fit the model
        """
        # first, re-configure the architecture.
        self.model = FeedforwardNetworkModule(self.hidden_dims, self.output_dim, self.activations,
                                              self.model_type, self.default_hidden_activation)
        self.model.configure_architecture(x.shape[-1])
        #self.model.layers=list()
        #self.model.activations=self.activations
        #self.model.configure_architecture(x.shape[-1])

        # now, fit the model
        iter = 0
        for epoch in range(self.num_epochs):
            # construct the batches

            input_data_loop_range = trange(x.shape[0])
            for i in range(input_data_loop_range):
                self.optimizer.zero_grad()  # zero the gradient buffers
                output = self.model(input)
                loss = self.criterion(output, y[i])
                loss.backward()
                self.optimizer.step()

                input_data_loop_range.set_postfix(loss=loss.item())

                # Print Loss
                #print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

            input_data_loop_range.close()






