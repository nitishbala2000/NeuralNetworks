import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidDerivative(z):
    t = sigmoid(z)
    return t * (1 - t)

def loss(yhat, y):
    return -np.sum(y * np.log(yhat))


def cost(AL, Y):
    m = Y.shape[1]
    return np.sum(loss(AL.T, Y.T)) / m

def softmax(z):
    t = np.exp(z)
    return t / np.sum(t, axis = 0, keepdims= True)


def tanh(z):
    return np.tanh(z)

def tanhDerivative(z):
    t = tanh(z)
    return 1 - t * t



def splitIntoBatches(X_train, Y_train, batchSize):
    m = X_train.shape[1]
    perm = np.random.permutation(m)
    shuffledX = X_train[:, perm]
    shuffledY = Y_train[:, perm]
    output = []
    for i in range(0, m, batchSize):
        batchX = shuffledX[:, list(range(i, i + batchSize))]
        batchY = shuffledY[:, list(range(i, i + batchSize))]
        output.append((batchX, batchY))

    return output

class Network(object):

    def __init__(self, n_x, hiddenLayerDimensions, n_y):
        self.num_layers = len(hiddenLayerDimensions) + 1
        layerDimensions = [n_x] + hiddenLayerDimensions + [n_y]
        self.W, self.b = {}, {}
        for i in range(1, len(layerDimensions)):
            self.W[i] = np.random.randn(layerDimensions[i], layerDimensions[i - 1]) * np.sqrt(1 / layerDimensions[i - 1])
            self.b[i] = np.zeros((layerDimensions[i], 1))

    def forwardPropagation(self, W, b, X):
        A, Z = {}, {}
        L = self.num_layers
        # Forward prop
        Z = {}
        A = {}
        A[0] = X
        for i in range(1, L):
            Z[i] = W[i] @ A[i - 1] + b[i]
            A[i] = tanh(Z[i])

        Z[L] = W[L] @ A[L - 1] + b[L]
        A[L] = softmax(Z[L])
        return A, Z


    def train(self, trainSetX, trainSetY, learning_rate = 0.5, epochs = 20, batchSize = 100):
        L = self.num_layers
        m = trainSetX.shape[1]

        while m % batchSize != 0:
            batchSize += 1

        totalBatches = m // batchSize
        for epoch in range(epochs):

            averageCost = 0
            for batchX, batchY in splitIntoBatches(trainSetX, trainSetY, batchSize):

                # Forward prop
                A, Z = self.forwardPropagation(self.W, self.b, batchX)

                costForThisBatch = cost(A[L], batchY)
                averageCost += costForThisBatch

                #Backward prop
                dZ, dW, dB = {}, {}, {}
                dZ[L] = A[L] - batchY

                for i in range(L, 0, -1):

                    dW[i] = (1/m) * (dZ[i] @ A[i - 1].T)
                    dB[i] = (1 / m) * np.sum(dZ[i], axis=1, keepdims=True)

                    if i > 1:
                        dZ[i - 1] = self.W[i].T @ dZ[i] * tanhDerivative(Z[i - 1])



                # update parameters
                for i in range(1, L + 1):
                    self.W[i] = self.W[i] - learning_rate * dW[i]
                    self.b[i] = self.b[i] - learning_rate * dB[i]

            averageCost /= totalBatches
            if epoch % 10 == 9:
                print("Epoch {} over. Average cost: {}".format(epoch + 1, averageCost))

        trainAccuracy = self.test(trainSetX, trainSetY)
        print("Training accuracy: " + str(trainAccuracy))


    def test(self, testsetX, testsetY):
        A, Z = self.forwardPropagation(self.W, self.b, testsetX)
        yhat = A[self.num_layers]

        m = testsetX.shape[1]
        totalCorrect = 0
        for i in range(m):
            prediction = np.argmax(yhat[: , i])
            correctResult = np.argmax(testsetY[:, i])
            if prediction == correctResult:
                totalCorrect += 1

        accuracy = totalCorrect / m
        return accuracy

    def predict(self, v):
        v = v.reshape(v.size, 1)
        A, Z = self.forwardPropagation(self.W, self.b, v)
        return np.argmax(A[self.num_layers])

'''
#MNIST- 96% accuracy reached
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images.T
Y_train = mnist.train.labels.T
X_test = mnist.test.images.T
Y_test = mnist.test.labels.T
net = Network(784, [50,50,50,50] , 10)
net.train(trainSetX= X_train, trainSetY= Y_train, epochs= 500, learning_rate= 1)
accuracy = net.test(X_test, Y_test)
print("Testing accuracy {}".format(accuracy))
'''


'''Belgian traffic signs- 87.5% test set accuracy reached
from TrafficSigns import load_data
def getIntoRequiredShape(images, labels):
    m = len(images)
    from skimage import transform
    images28 = [transform.resize(image, (50, 50)) for image in images]

    from skimage.color import rgb2gray
    images28 = np.array(images28)
    images28 = rgb2gray(images28)

    X = images28.reshape(m, 2500).T
    Y = np.zeros((62, m))
    for i, label in enumerate(labels):
        Y[label][i] = 1

    return X, Y

images, labels = load_data("BelgiumTSC_Training")
trainSetX, trainSetY = getIntoRequiredShape(images, labels)

net = Network(2500, [50,50,50,50], 62)
net.train(trainSetX, trainSetY, batchSize= 75, learning_rate= 0.3, epochs=1000)

images, labels = load_data("BelgiumTSC_Testing/Testing")
TestSetX, TestSetY = getIntoRequiredShape(images, labels)
accuracy = net.test(TestSetX, TestSetY)
print("Testing accuracy: " + str(accuracy))
'''

import h5py
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

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,_ = load_dataset()
train_set_y, test_set_y = getIntoRequiredForm(train_set_y), getIntoRequiredForm(test_set_y)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
n_x, n_y = num_px * num_px * 3, 1
train_set_x = train_set_x_orig.reshape((m_train, num_px * num_px * 3)).T/255.
test_set_x = test_set_x_orig.reshape((m_test, num_px * num_px * 3)).T/255.


net = Network(n_x, [50,50,50,50], 2)
net.train(train_set_x, train_set_y, learning_rate= 0.05, epochs= 2000, batchSize= m_train)
print("Testing accuracy: " + str(net.test(test_set_x, test_set_y)))
