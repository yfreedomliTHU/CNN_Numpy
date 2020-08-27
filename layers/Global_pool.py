import numpy as np
from module import Module

class Global_Avg_Pooling(Module):
    def __init__(self, input_shape):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        """
        self.input_shape = input_shape
        self.batchsize = input_shape[0]
        self.output_channels = input_shape[-1]
        self.output_shape = [self.batchsize, 1, 1, self.output_channels]
        self.gradient_index = np.ones(input_shape) / (self.input_shape[1] * self.input_shape[2])

    def forward(self, x):
        pool_out = np.zeros(self.output_shape)
        for batch_id in range(self.batchsize):
            for channel_id in range(self.output_channels):
                pool_out[batch_id, :, :, channel_id] = np.mean(x[batch_id, :, :, channel_id], keepdims=True)
        return pool_out
    
    def update_error(self):
        error_updated = np.zeros(self.input_shape)
        for batch_id in range(self.batchsize):
            for channel_id in range(self.output_channels):
                error_updated[batch_id, :, :, channel_id] = self.error[batch_id, 0, 0, channel_id]*self.gradient_index[batch_id, :, :, channel_id]
        return error_updated

    def SGD(self, error):
        self.error = error
        error_next = self.update_error() # get next error
        return error_next

    def backward(self, lr=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        pass

class Global_Max_Pooling(Module):
    def __init__(self, input_shape):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        """
        self.input_shape = input_shape
        self.batchsize = input_shape[0]
        self.output_channels = input_shape[-1]
        self.output_shape = [self.batchsize, 1, 1, self.output_channels]
        self.gradient_index = np.zeros(input_shape)

    def forward(self, x):
        pool_out = np.zeros(self.output_shape)
        for batch_id in range(self.batchsize):
            for channel_id in range(self.output_channels):
                poolmap = x[batch_id, :, :, channel_id]
                pool_out[batch_id, :, :, channel_id] = np.max(poolmap, keepdims=True)
                h_idx, w_idx = np.unravel_index(poolmap.argmax(), poolmap.shape)
                self.gradient_index[batch_id, h_idx, w_idx, channel_id] = 1 # save the index of maxnum(mask)
        return pool_out
    
    def update_error(self):
        error_updated = np.zeros(self.input_shape)
        for batch_id in range(self.batchsize):
            for channel_id in range(self.output_channels):
                error_updated[batch_id, :, :, channel_id] = self.error[batch_id, 0, 0, channel_id]*self.gradient_index[batch_id, :, :, channel_id]
        return error_updated

    def SGD(self, error):
        self.error = error
        error_next = self.update_error() # get next error
        return error_next

    def backward(self, lr=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        pass

    
if __name__ == "__main__":
    img = np.ones((2, 4, 4, 3))
    img[:,0,3,:] = 4
    pool = Global_Max_Pooling(img.shape)
    result = pool.forward(img)
    print(result.shape)
    print(result)
    next1 = result.copy() + 1
    error_next = pool.SGD(next1-result)
    print(error_next[:, 0, 3,:])
   
        