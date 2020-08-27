import numpy as np
from Dataset import DataLoader
import random
import time


def sigmoid(x):
    return np.longfloat(1.0 / (1.0 + np.exp(-x)))

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))

def vectorized_result(j):
    e = np.zeros((1, 10))
    e[0][j] = 1.0
    return e


class DNN():
    def __init__(self, dnn_shape, lr = 3):
        #dnn_shape:[input_nodes, hidden_nodes, output_nodes]
        self.lr = lr
        self.num_layers = len(dnn_shape)
        self.dnn_shape = dnn_shape
        # random init weighs and bias matrix
        self.bias = [np.random.randn(1, y) for y in dnn_shape[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(dnn_shape[:-1], dnn_shape[1:])]

    def predict(self, images):
        result = images
        for b, w in zip(self.bias, self.weights):
            result = sigmoid(np.dot(result, w) + b)
        return result

    def train(self, train_img_batch, train_res_batch):
        b_error, w_error = self.forward(train_img_batch, train_res_batch)
        self.update(b_error, w_error)


    def forward(self, training_image, training_result):
        data_lenth = len(training_image)
        batch_b_error = [np.zeros(b.shape) for b in self.bias]
        batch_w_error = [np.zeros(w.shape) for w in self.weights]
        for image, result in zip(training_image, training_result):
            b_error = [np.zeros(b.shape) for b in self.bias]
            w_error = [np.zeros(w.shape) for w in self.weights]
            out_data = [image]
            in_data = []
            for b, w in zip(self.bias, self.weights):
                in_data.append(np.dot(out_data[-1], w) + b)
                out_data.append(sigmoid(in_data[-1]))
            b_error[-1] = sigmoid_prime(in_data[-1]) * (out_data[-1] - result)
            w_error[-1] = np.dot(out_data[-2].transpose(), b_error[-1])
            for l in range(2, self.num_layers):
                b_error[-l] = sigmoid_prime(in_data[-l]) * \
                            np.dot(b_error[-l+1], self.weights[-l+1].transpose())
                w_error[-l] = np.dot(out_data[-l-1].transpose(), b_error[-l])

            batch_b_error = [bbe + be for bbe, be in zip(batch_b_error, b_error)]
            batch_w_error = [bwe + we for bwe, we in zip(batch_w_error, w_error)]
        b_error_mean = [bbe / data_lenth for bbe in batch_b_error]
        w_error_mean = [bwe / data_lenth for bwe in batch_w_error]
        return  b_error_mean, w_error_mean
        

    def update(self,b_error, w_error):
        self.bias = [b - self.lr * be for b, be in zip(self.bias, b_error)]
        self.weights = [w - self.lr * we for w, we in zip(self.weights, w_error)]


    def test(self, test_image, test_result):
        results = [(np.argmax(self.predict(image)), result)
                   for image, result in zip(test_image, test_result)]
        right = sum(int(x == y) for (x, y) in results)
        Acc = right / len(test_result)
        #print("Test Accuracy：{0}/{1}".format(right, len(test_result)))
        return Acc

    def save_weight(self):
        np.savez('./checkpoint/weights.npz', *self.weights)
        np.savez('./checkpoint/bias.npz', *self.bias)

    def load_weight(self):
        length = len(self.dnn_shape) - 1
        file_weights = np.load('./checkpoint/weights.npz')
        file_bias = np.load('./checkpoint/bias.npz')
        self.weights = []
        self.bias = []
        for i in range(length):
            index = "arr_" + str(i)
            self.weights.append(file_weights[index])
            self.bias.append(file_bias[index])
            
def get_minibatchs(data, label,batch_size):
    minibatch_data = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
    minibatch_label = [label[k:k+batch_size] for k in range(0, len(data), batch_size)]
    return minibatch_data, minibatch_label

if __name__ == "__main__":
    
    batch_size = 10
    DL = DataLoader()
    train_data, train_label0 = DL.load_mnist('./data/mnist')
    test_data, test_label = DL.load_mnist('./data/mnist', 't10k')

    train_images = [(im / 255).reshape(1, 784) for im in train_data] 
    test_images = [(im / 255).reshape(1, 784) for im in test_data] 
    train_label = [vectorized_result(int(i)) for i in train_label0]
    train_img_batchs, train_label_batchs = get_minibatchs(train_images, train_label, batch_size)

    model = DNN([28 * 28, 64, 10])
    steps = 0
    eval_freq = 6000
    for epoch in range(50):

        for train_img_batch, train_res_batch in zip(train_img_batchs, train_label_batchs):
            # normal SGD train
            # print("normal training!")
            model.train(train_img_batch, train_res_batch)
            steps += 1

        Acc = model.test(test_images, test_label)
        
        print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + " epoch:{} Test Accuracy:{}".format(epoch, Acc))



    
