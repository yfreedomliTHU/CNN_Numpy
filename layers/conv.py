import numpy as np
import math
from functools import reduce
from module import Module

class Conv2D(Module):
    def __init__(self, input_shape, output_channels, kernel_size=3, stride=1, init_params=True, padding = 0):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        output_channels:(int) the channel numbers of output
        kernel_size:(int) the kernel size of Conv
        stride:(int) the stride
        init_params:(bool) initialize statement of params of Conv2D
        """
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_channels = input_shape[-1]
        self.H_in = input_shape[1]
        self.W_in = input_shape[2]
        self.batchsize = input_shape[0]
        self.init_params = init_params
        self.padding = padding

        self.H_out = int((self.H_in - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.W_out = int((self.W_in - self.kernel_size + 2 * self.padding) / self.stride + 1)

        self.weights = np.random.randn(self.kernel_size, self.kernel_size, self.input_channels, self.output_channels)
        self.bias = np.random.randn(self.output_channels)

        if init_params:
            self.reset_parameters()

        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

        # outputshape = errorshape:[batchsize, H_out, W_out, C_out]
        self.error = np.zeros((self.batchsize, self.H_out, self.W_out, self.output_channels))
        self.output_shape = self.error.shape

    def reset_parameters(self):
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.input_shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            size=(self.kernel_size, self.kernel_size, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

    def forward(self, x):
        # Transfer to Col
        weights_col = self.weights.reshape(-1, self.output_channels)
        weights_col_batch = weights_col[np.newaxis, :].repeat(self.batchsize, axis=0) #repeat to batch
        self.col_img = self.im2col(x) # transfer Image to COL
        conv_out = np.reshape(np.matmul(self.col_img, weights_col_batch) + self.bias, self.error.shape)
        return conv_out

    def cal_grad(self, error_col):
        #calculate gradient
        for i in range(self.batchsize):
            self.weights_gradient += np.dot(self.col_img[i].T, error_col[i]).reshape(self.weights.shape)
        self.bias_gradient += np.sum(error_col, axis=(0, 1))
        self.weights_gradient /= self.batchsize
        self.bias_gradient /= self.batchsize

    def update_error(self):
        #deconv to get next error
        pad_error = np.pad(self.error, ((0, 0), (self.kernel_size - 1, self.kernel_size - 1), 
                        (self.kernel_size - 1, self.kernel_size - 1), (0, 0)),'constant', constant_values=0)
        weights_flip = self.weights[::-1, ::-1, :, :] #flip
        weights_flip = weights_flip.swapaxes(2, 3)    #swap
        weights_flip_col = weights_flip.reshape(-1, self.input_channels)
        pad_error_col = self.im2col(pad_error) #Transform error to col
        error_updated = np.reshape(np.dot(pad_error_col , weights_flip_col), self.input_shape)
        return error_updated


    def SGD(self, error):
        self.error = error
        error_col = np.reshape(error, [self.batchsize, -1, self.output_channels])
        self.cal_grad(error_col) #calculate gradient
        error_next = self.update_error() # get next error
        return error_next

    def backward(self, lr=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights -= lr * (1 - weight_decay) * self.weights_gradient
        self.bias -= lr * (1 - weight_decay) * self.bias_gradient

        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)
        

    def im2col(self, image):
         # image shape([batchsize, height, width, channel])
        image_col = []
        for i in range(0, image.shape[1] - self.kernel_size + 1, self.stride):
            for j in range(0, image.shape[2] - self.kernel_size + 1, self.stride):
                col = image[:, i:i + self.kernel_size, j:j + self.kernel_size, :].reshape(image.shape[0],-1)
                image_col.append(col)
        image_col = np.array(image_col)
        return image_col.transpose(1,0,2)
   
if __name__ == "__main__":
    # img = np.random.standard_normal((2, 32, 32, 3))
    img = np.ones((5, 32, 32, 3))
    img *= 2
    conv = Conv2D(img.shape, 12, 3, 1)
    next = conv.forward(img)
    next1 = next.copy() + 1
    conv.SGD(next1-next)
    print(conv.weights_gradient)
    print(conv.bias_gradient)
    conv.backward()
