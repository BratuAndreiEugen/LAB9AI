import numpy as np
from matplotlib import pyplot as plt

from ann.genericAnn import ANN
from loss.losses import my_cross_entropy_loss
from normalization.normal import statisticalNormalisation, getStatisticalParameters, statisticalScalingParam
from solvers.MNISTSolver import normalize
from utils.data import get_mnist
from sklearn.datasets import load_digits

from utils.photoLoader import getImageMatrices


def sepia():
    m1 = getImageMatrices("C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set\\normal", 4)
    l1 = np.array([np.array([1, 0]) for i in range(len(m1))])
    m2 = getImageMatrices("C:\Proiecte SSD\Python\LAB9AI\sepiaData\\training_set\sepia", 4)
    l2 = np.array([np.array([0, 1]) for i in range(len(m2))])
    m3 = getImageMatrices("C:\Proiecte SSD\Python\LAB9AI\sepiaData\\test_set\\normal", 4)
    l3 = np.array([np.array([1, 0]) for i in range(len(m3))])
    m4 = getImageMatrices("C:\Proiecte SSD\Python\LAB9AI\sepiaData\\test_set\sepia", 4)
    l4 = np.array([np.array([0, 1]) for i in range(len(m4))])

    images = np.concatenate((m1, m2, m3, m4))
    images = [np.concatenate(images[i]) for i in range(len(images))]
    labels = np.concatenate((l1, l2, l3, l4))
    print(type(images))
    print(images[0])


    # impartire test / train
    indexes = [i for i in range(len(images))]
    trainSample = np.random.choice(indexes, int(0.8 * len(images)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]
    trainImages = [images[i] for i in trainSample]
    trainLabels = [labels[i] for i in trainSample]
    validationImages = [images[i] for i in validationSample]
    validationLabels = [labels[i] for i in validationSample]

    # normalizare ( nu e neaparat necesar -- sterge pana la spatiu )
    # trainImages, param = normalize(trainImages)
    # validationImages, pr = normalize(validationImages, param)
    #
    # t = []
    # for img in trainImages:
    #     t.append(np.array(img))
    # trainImages = t
    # t = []
    # for img in validationImages:
    #     t.append(np.array(img))
    # validationImages = t

    trainOutputs = []
    for l in trainLabels:
        trainOutputs.append(np.argmax(l))

    outputNames = ['normal', 'sepia']
    plt.hist(trainOutputs, rwidth=0.8)
    plt.xticks(np.arange(len(outputNames)), outputNames)
    plt.show()

    # antrenare
    ann = ANN(trainImages, trainLabels, 2)  # inputs, outputs, neurons in hidden layer
    ann.train(learn_rate= 0.01, activation=ANN.softmax, epochs=200)

    # testare

    # accuracy and cross entropy loss
    actual = []
    for label in validationLabels:
        actual.append(label.argmax())

    predicted = []
    good = 0
    for i in range(len(validationImages)):
        img = validationImages[i]
        label = validationLabels[i]
        # plt.imshow(img.reshape(28, 28), cmap="Greys")
        img.shape += (1,)
        p = ann.predict([img])
        p = np.concatenate(p, axis=0).flatten()
        # print(p.argmax())
        # plt.show()
        if p.argmax() == label.argmax():
            good += 1

        predicted.append(p)

    print("Test Accuracy : " + str(good / len(validationImages)))
    print("Cross-entropy loss : " + str(my_cross_entropy_loss(actual, predicted)))

    img = validationImages[0]
    plt.imshow(img.reshape(4, 4), cmap="Greys")
    img.shape += (1,)
    plt.show()