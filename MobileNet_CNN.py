import numpy as np
from my_torch import Conv2D, Relu

# forward & backward & SGD
# MobileNet_CNN_kernel_sizexoutput_channel
class MobileNet_CNN_5x12():
    def __init__(self, input_shape_, output_channels_, k_size=5, stride_=1):
        """
        params:
        input_shape_:the shape of input data[batchsize, High, Width, Channel]
        output_channels_:(int) the channel numbers of output
        k_size:(int) the kernel size of Conv
        _stride:(int) the stride
        """
        # kernel_size = 5, input_channels = 1, output_channels = 12
        self.conv_5x12 = Conv2D(input_shape=input_shape_, output_channels=output_channels_, kernel_size=k_size, stride=stride_)
        self.output_shape = self.conv_5x12.output_shape

    def forward(self, x):
        conv_5x12_out = self.conv_5x12.forward(x)
        return conv_5x12_out

    def SGD(self, error):
        error_out = self.conv_5x12.SGD(error)
        return error_out

    def backward(self, lr_=0.00001, weight_decay_=0.0004):
        self.conv_5x12.backward(lr=lr_, weight_decay=weight_decay_)

class MobileNet_CNN_3x24():
    def __init__(self, input_shape_, output_channels_, k_size=3, stride_=1):
        """
        params:
        input_shape:the shape of input data[batchsize, High, Width, Channel]
        output_channels_:(int) the channel numbers of output
        k_size:(int) the kernel size of Conv
        _stride:(int) the stride
        """
        # kernel_size = 5, input_channels(M) = 12, output_channels(N) = 24
        self.in_shape = input_shape_
        self.dep_conv_shape = [input_shape_[0], input_shape_[1], input_shape_[2], 1] #every channel do dep_conv
        self.dep_conv1 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv2 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv3 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv4 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv5 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv6 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv7 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv8 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv9 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv10 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv11 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.dep_conv12 = Conv2D(input_shape=self.dep_conv_shape, output_channels=1, kernel_size=k_size, stride=stride_)
        self.depconvout_shape = [self.dep_conv1.output_shape[0], self.dep_conv1.output_shape[1], self.dep_conv1.output_shape[2], 12]
        self.depconvout = np.zeros(self.depconvout_shape)
        self.error_depconv = np.zeros(self.in_shape)
        self.relu = Relu(self.depconvout_shape)
        self.point_conv = Conv2D(input_shape=self.depconvout_shape, output_channels=output_channels_, kernel_size=1, stride=1)
        self.output_shape = self.point_conv.output_shape

    def forward(self, x):
        """
        x:input data, do depth_wise_conv every channel
        x.shape:[batchsize, H, W, C]
        """
        x = x[:,:,:,np.newaxis,:]
        #print(x.shape)
        #print(x[:,:,:,:,0].shape)
        self.depconvout[:,:,:,0] = self.dep_conv1.forward(x[:,:,:,:,0]).squeeze()
        self.depconvout[:,:,:,1] = self.dep_conv2.forward(x[:,:,:,:,1]).squeeze()
        self.depconvout[:,:,:,2] = self.dep_conv3.forward(x[:,:,:,:,2]).squeeze()
        self.depconvout[:,:,:,3] = self.dep_conv4.forward(x[:,:,:,:,3]).squeeze()
        self.depconvout[:,:,:,4] = self.dep_conv5.forward(x[:,:,:,:,4]).squeeze()
        self.depconvout[:,:,:,5] = self.dep_conv6.forward(x[:,:,:,:,5]).squeeze()
        self.depconvout[:,:,:,6] = self.dep_conv7.forward(x[:,:,:,:,6]).squeeze()
        self.depconvout[:,:,:,7] = self.dep_conv8.forward(x[:,:,:,:,7]).squeeze()
        self.depconvout[:,:,:,8] = self.dep_conv9.forward(x[:,:,:,:,8]).squeeze()
        self.depconvout[:,:,:,9] = self.dep_conv10.forward(x[:,:,:,:,9]).squeeze()
        self.depconvout[:,:,:,10] = self.dep_conv11.forward(x[:,:,:,:,10]).squeeze()
        self.depconvout[:,:,:,11] = self.dep_conv12.forward(x[:,:,:,:,11]).squeeze()
        relu_out = self.relu.forward(self.depconvout)
        point_conv_out = self.point_conv.forward(relu_out)
        return point_conv_out

    def SGD(self, error):
        error_point_conv = self.point_conv.SGD(error)
        error_relu = self.relu.SGD(error_point_conv)
        error_relu = error_relu[:,:,:,np.newaxis,:]
        self.error_depconv[:,:,:,0] = self.dep_conv1.SGD(error_relu[:,:,:,:,0]).squeeze()
        self.error_depconv[:,:,:,1] = self.dep_conv2.SGD(error_relu[:,:,:,:,1]).squeeze()
        self.error_depconv[:,:,:,2] = self.dep_conv3.SGD(error_relu[:,:,:,:,2]).squeeze()
        self.error_depconv[:,:,:,3] = self.dep_conv4.SGD(error_relu[:,:,:,:,3]).squeeze()
        self.error_depconv[:,:,:,4] = self.dep_conv5.SGD(error_relu[:,:,:,:,4]).squeeze()
        self.error_depconv[:,:,:,5] = self.dep_conv6.SGD(error_relu[:,:,:,:,5]).squeeze()
        self.error_depconv[:,:,:,6] = self.dep_conv7.SGD(error_relu[:,:,:,:,6]).squeeze()
        self.error_depconv[:,:,:,7] = self.dep_conv8.SGD(error_relu[:,:,:,:,7]).squeeze()
        self.error_depconv[:,:,:,8] = self.dep_conv9.SGD(error_relu[:,:,:,:,8]).squeeze()
        self.error_depconv[:,:,:,9] = self.dep_conv10.SGD(error_relu[:,:,:,:,9]).squeeze()
        self.error_depconv[:,:,:,10] = self.dep_conv11.SGD(error_relu[:,:,:,:,10]).squeeze()
        self.error_depconv[:,:,:,11] = self.dep_conv12.SGD(error_relu[:,:,:,:,11]).squeeze()
        return self.error_depconv

    def backward(self, lr_=0.00001, weight_decay_=0.0004):
        self.point_conv.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv12.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv11.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv10.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv9.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv8.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv7.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv6.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv5.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv4.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv3.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv2.backward(lr=lr_, weight_decay=weight_decay_)
        self.dep_conv1.backward(lr=lr_, weight_decay=weight_decay_)
        


        