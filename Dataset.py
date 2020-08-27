import struct
from glob import glob
import os
import numpy as np

class DataLoader():
    def __init__(self):
        self.load_status()

    def load_status(self):
        print("Loading Data...")
        return 0

    def check_path(self, path):
        if not os.path.exists(path):
            print("Error! File Path Not Exists!")
            exit(1)

    def Normlize(self, X):
        # Normalize the data
        X -= int(np.mean(X)) # subtract mean
        X /= int(np.std(X)) # divide by standard deviation
        return X 

    def load_mnist(self, path, mode='train', Norm = False):
        self.check_path(path)
        """Load MNIST data from `path`"""
        images_path = glob('./%s/%s*3-ubyte' % (path, mode))[0]
        labels_path = glob('./%s/%s*1-ubyte' % (path, mode))[0]

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                    lbpath.read(8))
            labels = np.fromfile(lbpath,
                                dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                imgpath.read(16))
            images = np.fromfile(imgpath,
                                dtype=np.uint8).astype(np.float32).reshape(len(labels), 784)

        if(Norm == True):
            images = self.Normlize(images)

        return images, labels

if __name__ == "__main__":
    DL = DataLoader()
    train_images, train_labels = DL.load_mnist('./data/mnist')
    test_images, test_labels = DL.load_mnist('./data/mnist', 't10k')

    print(train_images[0])
    