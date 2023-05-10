import numpy as np
from matplotlib import pyplot as plt

from ann.genericAnn import ANN
from loss.losses import my_cross_entropy_loss
from normalization.normal import statisticalNormalisation, getStatisticalParameters, statisticalScalingParam
from solvers.MNISTSolver import normalize
from utils.data import get_mnist
from sklearn.datasets import load_digits


def digits():
    print("\nsklearn DIGITS cod propriu (Poze 8x8 pixeli):")

    data = load_digits()
    outputs = []
    for op in data['target']:
        v = [0 for i in range(10)]
        v[op] = 1
        outputs.append(np.array(v))

    labels = np.array(outputs) # one hot
    images = data.images
    act = []
    for img in images:
        act.append(np.concatenate(img))
    images = act


    # impartire test / train
    np.random.seed(5)
    indexes = [i for i in range(len(images))]
    trainSample = np.random.choice(indexes, int(0.8 * len(images)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]
    trainImages = [images[i] for i in trainSample]
    trainLabels = [labels[i] for i in trainSample]
    validationImages = [images[i] for i in validationSample]
    validationLabels = [labels[i] for i in validationSample]

    # normalizare ( nu e neaparat necesar -- sterge pana la spatiu )
    trainImages, param = normalize(trainImages)
    validationImages, pr = normalize(validationImages, param)

    t = []
    for img in trainImages:
        t.append(np.array(img))
    trainImages = t
    t = []
    for img in validationImages:
        t.append(np.array(img))
    validationImages = t

    trainOutputs = []
    for l in trainLabels:
        trainOutputs.append(np.argmax(l))

    outputNames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.hist(trainOutputs, rwidth=0.8)
    plt.xticks(np.arange(len(outputNames)), outputNames)
    plt.show()

    # antrenare
    ann = ANN(trainImages, trainLabels, 10)  # inputs, outputs, neurons in hidden layer
    ann.train(activation=ANN.softmax, epochs=5)

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