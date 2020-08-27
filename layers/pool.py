import numpy as np
from module import Module

class MaxPooling2D(Module):
    def __init__(self, input_shape, kernel_size=2, stride=2):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        kernel_size:(int) the kernel size
        stride:(int) the stride
        """
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_channels = input_shape[-1]
        self.H_in = input_shape[1]
        self.W_in = input_shape[2]
        self.batchsize = input_shape[0]

        self.H_out = int((self.H_in - self.kernel_size) / self.stride + 1)
        self.W_out = int((self.W_in - self.kernel_size) / self.stride + 1)

        self.output_shape = [self.batchsize, self.H_out, self.W_out, self.output_channels]
        self.gradient_index = np.zeros(input_shape)

    def forward(self, x):
        pool_out = np.zeros(self.output_shape)
        for batch_id in range(self.batchsize):
            for channel_id in range(self.output_channels):
                for i_count, i in enumerate(range(0, self.H_in - self.kernel_size + 1, self.stride)):
                    for j_count, j in enumerate(range(0, self.W_in - self.kernel_size + 1, self.stride)):
                        poolmap = x[batch_id, i:i + self.kernel_size, j:j + self.kernel_size, channel_id]
                        pool_out[batch_id, i_count, j_count, channel_id] = np.max(poolmap)
                        h_idx, w_idx = np.unravel_index(poolmap.argmax(), poolmap.shape)
                        self.gradient_index[batch_id, i+h_idx, j+w_idx, channel_id] = 1 # save the index of maxnum(mask)
        return pool_out

    def update_error(self):
        error_updated = np.repeat(np.repeat(self.error, self.stride, axis=1), self.stride, axis=2) * self.gradient_index
        return error_updated

    def SGD(self, error):
        self.error = error
        error_next = self.update_error() # get next error
        return error_next

    def backward(self, lr=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        pass

class AvgPooling2D(Module):
    def __init__(self, input_shape, kernel_size=2, stride=2):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        kernel_size:(int) the kernel size
        stride:(int) the stride
        """
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_channels = input_shape[-1]
        self.H_in = input_shape[1]
        self.W_in = input_shape[2]
        self.batchsize = input_shape[0]

        self.H_out = int((self.H_in - self.kernel_size) / self.stride + 1)
        self.W_out = int((self.W_in - self.kernel_size) / self.stride + 1)

        self.output_shape = [self.batchsize, self.H_out, self.W_out, self.output_channels]
        self.gradient_index = np.ones(input_shape) / (self.kernel_size * self.kernel_size)

    def forward(self, x):
        pool_out = np.zeros(self.output_shape)
        for batch_id in range(self.batchsize):
            for channel_id in range(self.output_channels):
                for i_count, i in enumerate(range(0, self.H_in - self.kernel_size + 1, self.stride)):
                    for j_count, j in enumerate(range(0, self.W_in - self.kernel_size + 1, self.stride)):
                        poolmap = x[batch_id, i:i + self.kernel_size, j:j + self.kernel_size, channel_id]
                        pool_out[batch_id, i_count, j_count, channel_id] = np.mean(poolmap)
        return pool_out

    def update_error(self):
        error_updated = np.repeat(np.repeat(self.error, self.stride, axis=1), self.stride, axis=2) * self.gradient_index
        return error_updated

    def SGD(self, error):
        self.error = error
        error_next = self.update_error() # get next error
        return error_next

    def backward(self, lr=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        pass

if __name__ == "__main__":
    img = np.ones((1, 4, 4, 1))
    img[:,3,3,:] = 4
    maxpool = MaxPooling2D(img.shape, 2, 2)
    result = maxpool.forward(img)
    print(result)