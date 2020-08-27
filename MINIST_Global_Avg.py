import numpy as np
import os
import time
import pickle
from my_torch import Conv2D, Relu, LRelu, Linear, MaxPooling2D, AvgPooling2D, CrossEntropyLoss, Global_Avg_Pooling
from Dataset import DataLoader


class Model():
    def __init__(self, batch_size):
        self.conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
        self.relu1 = Relu(self.conv1.output_shape)
        self.maxpool1 = MaxPooling2D(self.relu1.output_shape)
        self.conv2 = Conv2D(self.maxpool1.output_shape, 24, 3, 1)
        self.relu2 = Relu(self.conv2.output_shape)
        self.maxpool2 = MaxPooling2D(self.relu2.output_shape)
        self.global_avg_pool = Global_Avg_Pooling(self.maxpool2.output_shape)
        self.fc = Linear(self.global_avg_pool.output_shape, 10)
        self.out = CrossEntropyLoss(self.fc.output_shape)

    def forward(self, x, label):
        # x shape [batchsize, H, W, C]
        # label shape[batchsize,]
        conv1_out = self.relu1.forward(self.conv1.forward(x))
        pool1_out = self.maxpool1.forward(conv1_out)
        conv2_out = self.relu2.forward(self.conv2.forward(pool1_out))
        pool2_out = self.maxpool2.forward(conv2_out)
        global_pool_out = self.global_avg_pool.forward(pool2_out)
        fc_out = self.fc.forward(global_pool_out)
        loss_out = self.out.forward(fc_out, label)
        return loss_out

    def SGD(self):
        error_out = self.out.SGD()
        self.conv1.SGD(self.relu1.SGD(self.maxpool1.SGD(
                        self.conv2.SGD(self.relu2.SGD(self.maxpool2.SGD(self.global_avg_pool.SGD(
                        self.fc.SGD(error_out))))))))

    def backward(self, learning_rate):
        self.fc.backward(lr=learning_rate, weight_decay=0.0004)
        self.conv2.backward(lr=learning_rate, weight_decay=0.0004)
        self.conv1.backward(lr=learning_rate, weight_decay=0.0004)

def learning_rate_exponential_decay(learning_rate, global_step, decay_rate=0.1, decay_steps=5000):
    '''
    Applies exponential decay to learning rate
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step/decay_steps)
    :return: learning rate decayed by step
    '''
    decayed_learning_rate = learning_rate * pow(decay_rate , float(global_step/decay_steps))
    return decayed_learning_rate
    
def train(model, img, label, batch_size, learning_rate):
    acc = 0
    loss = model.forward(img, np.array(label))
    for j in range(batch_size):
        if np.argmax(model.out.predict_result[j]) == label[j]:
            acc += 1
    model.SGD()
    model.backward(learning_rate)
    return acc / batch_size, loss

def test(model, test_images, test_labels, batch_size):
    # validation
    val_loss = 0
    val_acc = 0
    batch_num = test_images.shape[0] // batch_size
    for i in range(batch_num):
        img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]

        val_loss += model.forward(img, np.array(label))
        
        for j in range(batch_size):
            if np.argmax(model.out.predict_result[j]) == label[j]:
                val_acc += 1

    return val_acc / (batch_num * batch_size), val_loss / batch_num


if __name__ == "__main__":
    logpath = 'logs'
    if not os.path.exists(logpath):
        os.mkdir(logpath)
    logdir = logpath + '/GlobalAvgPool_log.txt'
    print_freq = 50
    val_freq = 200
    DL = DataLoader()
    images, labels = DL.load_mnist('./data/mnist')
    test_images, test_labels = DL.load_mnist('./data/mnist', 't10k')
    batch_size = 100
    model = Model(batch_size)
    #record 
    train_loss_record = []
    train_acc_record = []
    val_loss_record = []
    val_acc_record = []
    with open(logdir, 'w') as logf:
        for epoch in range(20):
            # save record every epoch
            history = dict()
            history['train_acc'] = train_acc_record
            history['train_loss'] = train_loss_record
            history['val_acc'] = val_acc_record
            history['val_loss'] = val_loss_record
            with open("GlobalAvgPool.pickle", "wb") as fp:   #Pickling
                pickle.dump(history, fp)      
           
            # random shuffle
            order = np.arange(images.shape[0])
            np.random.shuffle(order)
            train_images = images[order]
            train_labels = labels[order]
        
            learning_rate = learning_rate_exponential_decay(5e-3, epoch, 0.1, 10)

            train_loss = 0
            train_acc = 0
            
            for i in range(train_images.shape[0] // batch_size):
                img = train_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
                label = train_labels[i * batch_size:(i + 1) * batch_size]
                tmp_acc, tmp_loss = train(model, img, label, batch_size, learning_rate)

                train_acc += tmp_acc
                train_loss += tmp_loss
                
                if (i+1) % print_freq == 0:
                    loginfo = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                        "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch,
                            i, tmp_acc, tmp_loss, learning_rate)
                    logf.write(loginfo + "\n")
                    print(loginfo)
                      
                if (i+1) % val_freq == 0 :
                    val_acc, val_loss = test(model, test_images, test_labels, batch_size)
                    #save
                    train_acc_record.append(train_acc)
                    train_loss_record.append(train_loss)
                    val_acc_record.append(val_acc)
                    val_loss_record.append(val_loss)
                    loginfo = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + "  epoch: %5d , train_acc: %.4f  train_loss: %.4f  val_acc: %.4f  val_loss: %.4f" % (
                                    epoch, train_acc / val_freq, train_loss / val_freq, val_acc, val_loss)
                    logf.write(loginfo + "\n")
                    print(loginfo)
                    train_loss = 0
                    train_acc = 0

            

        