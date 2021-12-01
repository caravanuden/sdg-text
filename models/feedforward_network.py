"""
This file implements a basic feedforward network which will take in inputs from.
Note: we can't set the input dimension until we call the fit function (because the dimsensionality of different
inputs differs; for some it's 300 dimensional, for others 384 dimensional.)
"""
from typing import List
import torch
import torch.nn as nn
from experiments.experiment import ModelType


class FeedforwardNN(nn.Module):
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
        :param num_epochs: the number of training epochs to use when fitting the model to the data.

        NOTE: the input dimension must be inferred when fit() is called.
        """
        super(FeedforwardNN, self).__init__()
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        #self.hidden_layer_activation = activation
        self.activations = activations
        self.model_type = model_type
        self.default_hidden_activation = default_hidden_activation

        # self.layers is what we'll use when we call forward.
        self.layers = list()
        self.architecture_configured = False

    def configure_architecture(self, input_dim):
        self.architecture_configured = True

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

        """
        if len(self.hidden_dims) == 0:
            self.layers = [nn.Linear(input_dim, self.output_dim)]
        else:
            for i in range(len(self.hidden_dims)):
                if i == 0:
                    self.layers.append(nn.Linear(input_dim, self.hidden_dims[i]))
                else:
                    self.layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))

        """


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


    def fit(self,x,y):
        """

        :param x: the inputs of shape (num_inputs, input_shape)
        :param y: their corresponding labels of shape (num_inputs,)
        :return: nothing. just fit the model
        """
        # first, configure the architecture.
        self.configure_architecture(x.shape[-1])

        # now, fit the model

        iter = 0
        for epoch in range(self.num_epochs):
            for i, input in enumerate(x):
                optimizer.zero_grad()  # zero the gradient buffers
                output = net(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()  # Does the update


            for i, (images, labels) in enumerate(train_loader):
                # Load images with gradient accumulation capabilities
                images = images.view(-1, 28 * 28).requires_grad_()

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = model(images)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                iter += 1

                if iter % 500 == 0:
                    # Calculate Accuracy
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for images, labels in test_loader:
                        # Load images with gradient accumulation capabilities
                        images = images.view(-1, 28 * 28).requires_grad_()

                        # Forward pass only to get logits/output
                        outputs = model(images)

                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted == labels).sum()

                    accuracy = 100 * correct / total

                    # Print Loss
                    print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))




