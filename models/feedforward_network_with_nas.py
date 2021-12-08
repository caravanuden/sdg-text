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
    def __init__(self, input_dim, hidden_dims_values_to_try: List[List[int]]=None, output_dim=1, activations: List[object]=list(),
                 model_type: ModelType = ModelType.classification, default_hidden_activation=nn.Sigmoid()):
        """
        :param input_dim: the input data dimension
        :param hidden_dims_values_to_try: list of lists of ints.
        :param output_dim
        :param activatiosn: the activations to be used in each hidden layer of the network; activations[i] is the
        activation to be used at layer i
        :param default_hidden_activation: this is used only if activations list is not specified.
        In that case, we just use the default activation at each layer.
        :param output_dim: the number of outputs. In our case, as long as we're only doing binary classification
        and regression, this should always be 1. But, if we switch to multi-class, then it could be more than one.


        NOTE: the input dimension must be inferred when fit() is called.
        """
        super(FeedforwardNetworkModuleForNAS, self).__init__()

        NUM_HIDDEN_UNITS_LIST = [50,100,150,200,250,input_dim]
        NUM_HIDDEN_UNITS_LIST_WITH_ZERO = [0, 50,100,150,200,250,input_dim]
        HIDDEN_ACTIVATIONS = [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()]

        self.hidden_dims_1 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST)
        self.hidden_dims_2 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)
        self.hidden_dims_3 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)
        self.hidden_dims_4 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)
        self.hidden_dims_5 = nn.ValueChoice(NUM_HIDDEN_UNITS_LIST_WITH_ZERO)


        self.fc1 = self.get_fc_block([self.hidden_dims_1, input_dim], HIDDEN_ACTIVATIONS)
        self.fc2 = self.get_fc_block([self.hidden_dims_2, self.hidden_dims_1], HIDDEN_ACTIVATIONS)
        self.fc3 = self.get_fc_block([self.hidden_dims_3, self.hidden_dims_2], HIDDEN_ACTIVATIONS)
        self.fc4 = self.get_fc_block([self.hidden_dims_4, self.hidden_dims_3], HIDDEN_ACTIVATIONS)

        # we need four of these to be able to select between the layer architectures.
        # see the forward function and self.input_choice
        self.final_layer_1 = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dims_1),
            nn.Sigmoid() # apply sigmoid to the output to get the probability
        )
        self.final_layer_2 = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dims_2),
            nn.Sigmoid()  # apply sigmoid to the output to get the probability
        )
        self.final_layer_3 = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dims_3),
            nn.Sigmoid()  # apply sigmoid to the output to get the probability
        )
        self.final_layer_4 = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dims_4),
            nn.Sigmoid()  # apply sigmoid to the output to get the probability
        )

        # for choosing between number of layers in architecture
        self.input_choice = nn.InputChoice(n_candidates=4, choose_from=["one layer","two layers","three layers","four layers"],
                                           n_chosen=1, reduction="none")


        # for now, just assume it's classification
        if model_type == ModelType.regression:
            raise NotImplementedError("NAS model can only do classification right now")



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
        output1 = self.fc1(x)
        output1 = self.final_layer_1(output1)

        output2 = self.fc2(output1)
        output2 = self.final_layer_2(output2)

        output3 = self.fc3(output2)
        output3 = self.final_layer_3(output3)

        output4 = self.fc4(output3)
        output4 = self.final_layer_4(output4)

        return self.input_choice(output1, output2, output3, output4)



"""
NOTE: we don't even need the interface for NAS because the framework is so good!
"""
class FeedforwardNewtorkForNAS(ModelInterface):
    # for now, this will be pretty bare-bones. I'm only going to allow hyperparams as inputs; we'll just let the NAS do
    # the work.
    def __init__(self, batch_size: int = 32, num_epochs: int = 10, optimizer: object = torch.optim.SGD,
                 criterion: object = None, learning_rate: float = 0.05):
        """
        For description of parameters not listed here, see FeedforwardNetworkModule

        :param batch_size: the batch size for trainig
        :param num_epochs: the number of training epochs to use when fitting the model to the data.
        :param optimizer: optimizer to use for training
        :param criterion: the type of loss to use
        :param learning_rate: the learning rate to use for training.
        """
        model_type = ModelType.classification

        self.user_specified_optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        if criterion is None:
            if model_type == ModelType.classification:
                self.criterion = torch.nn.BCELoss()
            elif model_type == ModelType.regression:
                raise NotImplementedError("NAS doesn't support regression right now.")
            else:
                raise NotImplementedError

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

        # now, fit the model
        mini_batch_range = trange(x.shape[0] // self.batch_size + 1)
        for i in mini_batch_range:
            end_range = min((i + 1) * self.batch_size, x.shape[0])

            # get batches
            batch_X = x[i*self.batch_size : end_range]
            batch_y = y[i*self.batch_size : end_range]
            batch_X = torch.from_numpy(batch_X).float()
            batch_y = torch.from_numpy(batch_y.reshape(-1,1)).float()

            # get the loss and do backprop
            self.optimizer.zero_grad()  # zero the gradient buffers
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
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
        probs = np.copy(output_probs)
        probs[probs<0.5] = 0
        probs[probs>=0.5] = 1
        return probs
