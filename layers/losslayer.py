import math
import numpy as np
from module import Module
from functools import reduce

class CrossEntropyLoss(Module):
    def __init__(self, input_shape):
        """
        params:
        input_shape:the shape of input data[batchsize, class_num]
        """
        self.softmax = np.zeros(input_shape)
        self.error = np.zeros(input_shape)
        self.batchsize = input_shape[0]
        self.class_num = input_shape[-1]
       
    def get_one_hot(self, target):
        target_one_hot = np.zeros((self.batchsize, self.class_num))
        for i in range(self.batchsize):
            index = target[i]
            target_one_hot[i, index] = 1
        return target_one_hot

    def predict(self, x):
        x_exp = np.zeros(x.shape)
        x_softmax = np.zeros(x.shape) #predict result via softmax
        for i in range(self.batchsize):
            x_exp[i, :] = np.exp(x[i, :]-np.max(x[i, :])) # norm & exp
            x_softmax[i, :] = x_exp[i, :] / np.sum(x_exp[i, :])

        self.predict_result = x_softmax
        
    def cal_loss(self):
        loss = 0
        delta = 1e-7
        for i in range(self.batchsize):
            t = self.target[i]
            y = self.predict_result[i]
            loss -= np.sum(t * np.log(y + delta))

        return loss / self.batchsize


    def forward(self, x, target):
        """
        forward to calulate loss
        params:
        target:True Label shape:[batchszie, 1]
        x:input_feature shape:[batchsize, class_num]
        """
        self.target = self.get_one_hot(target)
        self.predict(x)
        loss = self.cal_loss()
        return loss

    def SGD(self):
        # get error via SGD
        error = self.predict_result - self.target
        return error

