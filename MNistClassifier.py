#MNIST- 96% accuracy reached
from tensorflow.examples.tutorials.mnist import input_data
import NNUsingTensorflow
import NNFromScratch

use_tensorflow = False
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images.T
Y_train = mnist.train.labels.T
X_test = mnist.test.images.T
Y_test = mnist.test.labels.T

if use_tensorflow:
    net = NNUsingTensorflow.Network(784, [50,50,50,50] , 10)
else:
    net = NNFromScratch.Network(784, [50,50,50,50] , 10)

net.train(trainSetX= X_train, trainSetY= Y_train, epochs= 500, learning_rate= 1)
accuracy = net.test(X_test, Y_test)
print("Testing accuracy {}".format(accuracy))