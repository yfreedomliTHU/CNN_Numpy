import math
import numpy as np
from module import Module
from functools import reduce

class Linear(Module):
    def __init__(self, input_shape, output_dim=2, init_params=True):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        output_dim:(int) the dim of output(class num)
        init_params:(bool) initialize statement of params
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.batchsize = input_shape[0]
        # calculate dim of input data
        self.input_feature_num = reduce(lambda x, y: x * y, input_shape[1:])

        self.weights = np.random.randn(self.input_feature_num, self.output_dim)
        self.bias = np.random.randn(self.output_dim)

        if init_params:
            self.reset_parameters()

        self.output_shape = [self.batchsize, self.output_dim]
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

    def reset_parameters(self):
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.input_shape) / self.output_dim)
        self.weights = np.random.standard_normal(size=(self.input_feature_num, self.output_dim)) / weights_scale
        self.bias = np.random.standard_normal(self.output_dim) / weights_scale

    def forward(self, x):
        self.x = x.reshape(self.batchsize, -1)
        linear_out = np.matmul(self.x, self.weights) + self.bias
        return linear_out

    def cal_grad(self, error):
        #calculate gradient
        self.weights_gradient = (np.dot(np.transpose(error), self.x)).T
        self.bias_gradient = np.sum(np.transpose(error), axis=1)
        self.weights_gradient /= self.batchsize
        self.bias_gradient /= self.batchsize

    def update_error(self):
        #deconv to get next error
        error_updated = np.reshape(np.dot(self.error, self.weights.T), self.input_shape)
        return error_updated

    def SGD(self, error):
        self.error = error
        self.cal_grad(error) #calculate gradient
        error_next = self.update_error() # get next error
        return error_next

    def backward(self, lr=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights -= lr * (1 - weight_decay) * self.weights_gradient
        self.bias -= lr * (1 - weight_decay) * self.bias_gradient

        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 4, 6, 5, 6, 3, 2, 1]])
    fc = Linear(img.shape, 2)
    out = fc.forward(img)

    fc.SGD(np.array([[1, -1],[3,2]]))

    print(fc.weights_gradient)
    print(fc.bias_gradient)

    fc.backward()
    #print(fc.weights)
