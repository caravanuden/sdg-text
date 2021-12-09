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
        self.__0 = torch.nn.modules.linear.Linear(in_features=384, out_features=32)
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
        self.__0 = torch.nn.modules.linear.Linear(in_features=32, out_features=256)
        self.__1 = _model__fc2__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__fc3__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice__mutation_8_2 = torch.nn.modules.activation.Sigmoid()

    def forward(self, *_inputs):
        layerchoice__mutation_8_2 = self.layerchoice__mutation_8_2(_inputs[0])
        return layerchoice__mutation_8_2



class _model__fc3(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=256, out_features=256)
        self.__1 = _model__fc3__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__fc4__1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerchoice__mutation_9_0 = torch.nn.modules.activation.ReLU()

    def forward(self, *_inputs):
        layerchoice__mutation_9_0 = self.layerchoice__mutation_9_0(_inputs[0])
        return layerchoice__mutation_9_0



class _model__fc4(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=256, out_features=64)
        self.__1 = _model__fc4__1()

    def forward(self, input__1):
        __0 = self.__0(input__1)
        __1 = self.__1(__0)
        return __1



class _model__final_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.modules.linear.Linear(in_features=64, out_features=2)
        self.__1 = torch.nn.modules.activation.Softmax()

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
        __Constant1 = 2
        __Constant2 = 3
        __Constant3 = 4
        __Attr7 = 4
        __Attr13 = 4
        __Attr19 = 4
        __fc1 = self.__fc1(x__1)
        __aten__ge8 = (__Attr7 >= __Constant1)
        __aten__ge14 = (__Attr13 >= __Constant2)
        __aten__ge20 = (__Attr19 >= __Constant3)
        __fc2 = self.__fc2(__fc1)
        __noop_identity12 = __fc2
        __fc3 = self.__fc3(__noop_identity12)
        __noop_identity18 = __fc3
        __fc4 = self.__fc4(__noop_identity18)
        __noop_identity24 = __fc4
        __final_layer = self.__final_layer(__noop_identity24)
        return __final_layer