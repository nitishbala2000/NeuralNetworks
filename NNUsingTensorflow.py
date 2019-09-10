import numpy as np
import tensorflow as tf


def splitIntoBatches(trainSetX, trainSetY, batchSize):
    m = trainSetX.shape[1]
    perm = np.random.permutation(m)
    shuffled_X = trainSetX[:, perm]
    shuffled_Y = trainSetY[:, perm]
    output = []
    for i in range(0,m,batchSize):
        batch_x = shuffled_X[: , list(range(i, i + batchSize))]
        batch_y = shuffled_Y[: , list(range(i, i + batchSize))]
        output.append((batch_x, batch_y))

    return output

class Network(object):

    def __init__(self, n_x, hiddenLayerDimensions, n_y):
        self.n_x = n_x
        self.n_y = n_y
        self.num_layers = len(hiddenLayerDimensions) + 1
        layerDimensions = [n_x] + hiddenLayerDimensions + [n_y]
        self.W, self.b = {}, {}
        self.sess = tf.Session()
        for i in range(1, len(layerDimensions)):
            self.W[i] = tf.Variable(tf.random_normal([layerDimensions[i], layerDimensions[i-1]], stddev=0.03), name='W'+str(i))
            self.b[i] = tf.Variable(tf.random_normal([layerDimensions[i],1]), name='b'+str(i))

    def train(self, trainSetX, trainSetY, batchSize = 100, learning_rate = 0.5, epochs = 10):
        L = self.num_layers
        m = trainSetX.shape[1]
        while m % batchSize != 0:
            batchSize += 1

        X = tf.placeholder(dtype= tf.float32, shape=[self.n_x, None])
        Y = tf.placeholder(dtype = tf.float32, shape = [self.n_y, None])
        A = X
        for i in range(1, L):
            A = tf.nn.relu(self.W[i] @ A + self.b[i])

        Z = self.W[L] @ A + self.b[L]
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(Z), labels=tf.transpose(Y)))
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        correct_prediction = tf.equal(tf.argmax(Y, axis=0), tf.argmax(Z, axis=0))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.sess.run(init)

        totalBatches = m // batchSize
        for epoch in range(epochs):
            averageCost = 0
            for batchX, batchY in splitIntoBatches(trainSetX, trainSetY, batchSize):
                _, c = self.sess.run([optimiser, cost], feed_dict={X:batchX, Y:batchY})
                averageCost += c

            averageCost /= totalBatches

            if epoch % 10 == 9:
                print("Cost after epoch {}: {}".format(epoch + 1, c))

        acc = self.sess.run(accuracy, feed_dict={X:trainSetX, Y:trainSetY})
        print("Finished training. Training accuracy: " + str(acc))


    def test(self, testsetX, testsetY):
        L = self.num_layers
        X = tf.placeholder(dtype= tf.float32, shape = [self.n_x, None])
        Y = tf.placeholder(dtype= tf.float32, shape = [self.n_y, None])

        A = X
        for i in range(1, L):
            Z = self.W[i] @ A + self.b[i]
            A = tf.nn.relu(Z)

        Z = self.W[L] @ A + self.b[L]
        correct_prediction = tf.equal(tf.argmax(Y, axis=0), tf.argmax(Z, axis=0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.sess.run(accuracy, feed_dict= {X: testsetX, Y:testsetY})

    def classify(self, v):
        v = v.reshape(self.n_x, 1)

        L = self.num_layers
        X = tf.placeholder(dtype=tf.float32, shape=[self.n_x, None])

        A = X
        for i in range(1, L):
            Z = self.W[i] @ A + self.b[i]
            A = tf.nn.relu(Z)

        Z = self.W[L] @ A + self.b[L]
        Z = self.sess.run(Z, feed_dict= {X : v})
        return np.argmax(Z, axis = 0)[0]
