import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nni.retiarii.nn.pytorch

import torch


class _model__fc1__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice__mutation_6_0 = torch.nn.modules.activation.ReLU()

    def forward(self, *_inputs):
        layerchoice__mutation_6_0 = self.layerchoice__mutation_6_0(_inputs[0])
        return layerchoice__mutation_6_0



class _model__fc1(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=384, out_features=100)
        self.__1 = _model__fc1__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__fc2__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice__mutation_7_3 = torch.nn.modules.activation.Tanh()

    def forward(self, *_inputs):
        layerchoice__mutation_7_3 = self.layerchoice__mutation_7_3(_inputs[0])
        return layerchoice__mutation_7_3



class _model__fc2(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=100, out_features=250)
        self.__1 = _model__fc2__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__fc3__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice__mutation_8_0 = torch.nn.modules.activation.ReLU()

    def forward(self, *_inputs):
        layerchoice__mutation_8_0 = self.layerchoice__mutation_8_0(_inputs[0])
        return layerchoice__mutation_8_0



class _model__fc3(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=250, out_features=250)
        self.__1 = _model__fc3__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__fc4__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice__mutation_9_1 = torch.nn.modules.activation.LeakyReLU()

    def forward(self, *_inputs):
        layerchoice__mutation_9_1 = self.layerchoice__mutation_9_1(_inputs[0])
        return layerchoice__mutation_9_1



class _model__fc4(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=250, out_features=150)
        self.__1 = _model__fc4__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__final_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=150, out_features=1)
        self.__1 = torch.nn.modules.activation.Sigmoid()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model(nn.Module):
    def __init__(self):
        super().__init__()
        self.__fc1 = _model__fc1()
        self.__fc2 = _model__fc2()
        self.__fc3 = _model__fc3()
        self.__fc4 = _model__fc4()
        self.__final_layer = _model__final_layer()

    def forward(self, x__1):
        __fc1 = self.__fc1(x__1)
        __fc2 = self.__fc2(__fc1)
        __fc3 = self.__fc3(__fc2)
        __fc4 = self.__fc4(__fc3)
        __final_layer = self.__final_layer(__fc4)
        return __final_layer