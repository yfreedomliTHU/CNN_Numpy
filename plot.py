import matplotlib.pyplot as plt
import numpy as np
import pickle


def train_plot(train_acc, train_loss, val_acc, val_loss):
    plt.figure
    epoch = np.arange(len(train_acc))
    plt.plot(epoch,train_acc,color='b',linestyle='-')
    plt.plot(epoch,train_loss,color='g', linestyle='-')
    plt.plot(epoch,val_acc,color='r',linestyle='-')
    plt.plot(epoch,val_loss,color='y', linestyle='-')
    plt.title('Acc/Loss - Iterations(200batch) curve')
    plt.xlabel('Iterations')
    plt.ylim(0,1)
    plt.ylabel('Acc/Loss')
    plt.legend(['Train_Acc','Train_Loss', 'Val_Acc', 'Val_Loss'],loc='best')
    plt.show()


if __name__ == "__main__":
    with open("LRELU_0001.pickle", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    train_acc = np.array(data['train_acc']) / 200
    train_loss = np.array(data['train_loss']) / 200
    val_acc = np.array(data['val_acc'])
    val_loss = np.array(data['val_loss'])
    train_plot(train_acc, train_loss, val_acc, val_loss)