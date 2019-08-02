import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images.T
Y_train = mnist.train.labels.T

X_test = mnist.test.images.T
Y_test = mnist.test.labels.T


learning_rate = 0.5
epochs = 10
batch_size = 100

x = tf.placeholder(tf.float32, [784, None])
y = tf.placeholder(tf.float32, [10, None])

W1 = tf.Variable(tf.random_normal([300, 784], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300,1]), name='b1')

W2 = tf.Variable(tf.random_normal([10, 300], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10,1]), name='b2')

z1 = tf.add(tf.matmul(W1, x), b1)
a1 = tf.nn.relu(z1)

z2 = tf.add(tf.matmul(W2, a1), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= tf.transpose(z2), labels= tf.transpose(y)))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, axis= 0), tf.argmax(z2, axis= 0))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:

   sess.run(init)
   total_batch = len(mnist.train.labels) // batch_size
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            batch_x = batch_x.T
            batch_y = batch_y.T

            _, c = sess.run([optimiser, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))


   print(sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))
