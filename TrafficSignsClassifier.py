import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
import os
import NNFromScratch
import NNUsingTensorflow


#Belgian traffic signs- 87.5% test set accuracy reached
def load_data(directory):
    X, Y = [], []
    for label in os.listdir(directory):
        if os.path.isdir(f"{directory}/{label}"):
            for file in os.listdir(f"{directory}/{label}"):
                if file.endswith("ppm"):

                    img = imread(f"{directory}/{label}/{file}")
                    img = rgb2gray(img)
                    img = resize(img, (50,50))

                    currentImage = []
                    for row in img:
                        currentImage += list(row)

                    X.append(currentImage)

                    lbl = np.zeros(62, dtype="float")
                    lbl[int(label)] = 1.0
                    Y.append(lbl)

    X = np.array(X).transpose()
    Y = np.array(Y).transpose()

    return X, Y


use_tensorflow = True


trainSetX, trainSetY = load_data("BelgiumTSC_Training")

if not use_tensorflow:
    net = NNFromScratch.Network(2500, [50,50,50,50], 62)
else:
    net = NNUsingTensorflow.Network(2500, [50,50,50,50], 62)


net.train(trainSetX, trainSetY, batchSize= 75, learning_rate= 0.3, epochs=1000)

TestSetX, TestSetY = load_data("BelgiumTSC_Testing/Testing")
accuracy = net.test(TestSetX, TestSetY)
print("Testing accuracy: " + str(accuracy))