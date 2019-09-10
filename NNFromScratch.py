import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
import os

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
                print("Epoch {} over. Average cost: {:.5f}".format(epoch + 1, averageCost))

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

    def classify(self, v):
        v = v.reshape(v.size, 1)
        A, Z = self.forwardPropagation(self.W, self.b, v)
        return np.argmax(A[self.num_layers])

