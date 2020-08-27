import numpy as np
from abc import abstractmethod

class Module(object):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def SGD(self):
        pass

    @abstractmethod
    def cal_grad(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def update_error(self):
        pass
