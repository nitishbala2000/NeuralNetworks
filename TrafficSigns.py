import os
import skimage
import tensorflow as tf
import numpy as np

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

if __name__ == '__main__':
    train_data_directory = os.path.normpath("BelgiumTSC_Training")
    test_data_directory = os.path.normpath("BelgiumTSC_Testing/Testing")

    images, labels = load_data(train_data_directory)


    from skimage import transform
    images28 = [transform.resize(image, (28,28)) for image in images]

    from skimage.color import rgb2gray
    images28 = np.array(images28)
    images28 = rgb2gray(images28)
    #images28 is of shape (4575,28,28)

    x = tf.placeholder(dtype = tf.float32, shape = [None, 28,28])
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    images_flat = tf.contrib.layers.flatten(x)

    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                        logits = logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correct_pred = tf.argmax(logits, 1)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)

    tf.set_random_seed(1234)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for i in range(201):
            print('EPOCH', i)
            _, accuracy_val, L = sess.run([train_op, accuracy, loss], feed_dict={x: images28, y: labels})
            if i % 10 == 0:
                print("Loss: ", L)
            print('DONE WITH EPOCH')


    #Evaluating performance
    test_images, test_labels = load_data(test_data_directory)
    test_images28 =[transform.resize(image, (28,28)) for image in test_images]
    test_images28 = rgb2gray(np.array(test_images28))
    predicted = sess.run([correct_pred], feed_dict= {x : test_images28})[0]
    correct = 0
    for i in range(len(predicted)):
        predictedValue = predicted[i]
        trueValue = test_labels[i]
        if predictedValue == trueValue:
            correct += 1

    print("Accuracy: {}".format(correct / len(predicted)))
    sess.close()
