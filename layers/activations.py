import numpy as np
from module import Module

class Relu(Module):
    def __init__(self, shape):
        self.error = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def SGD(self, error):
        self.error = error
        self.error[self.x<0]=0
        return self.error

    def backward(self):
        pass

class LRelu(Module):
    def __init__(self, shape, alpha = 0.001):
        self.error = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0) + self.alpha * np.minimum(x, 0)

    def SGD(self, error):
        self.error = error
        self.error[self.x<=0] *= self.alpha
        return self.error

    def backward(self):
        pass
