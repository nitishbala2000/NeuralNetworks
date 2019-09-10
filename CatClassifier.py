
import h5py
import numpy as np
import NNFromScratch
import NNUsingTensorflow

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def getIntoRequiredForm(y):
    output = np.zeros((2, y.size))
    for i in range(len(y[0])):
        label = y[0][i]
        output[label][i] = 1

    return output

use_tensorflow = False

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,_ = load_dataset()


train_set_y, test_set_y = getIntoRequiredForm(train_set_y), getIntoRequiredForm(test_set_y)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
n_x, n_y = num_px * num_px * 3, 1
train_set_x = train_set_x_orig.reshape((m_train, num_px * num_px * 3)).T/255
test_set_x = test_set_x_orig.reshape((m_test, num_px * num_px * 3)).T/255

if use_tensorflow:
    net = NNUsingTensorflow.Network(n_x, [50,50,50,50], 2)
else:
    net = NNFromScratch.Network(n_x, [50,50,50,50], 2)

net.train(train_set_x, train_set_y, learning_rate= 0.05, epochs= 2000, batchSize= m_train)
print("Testing accuracy: " + str(net.test(test_set_x, test_set_y)))