import numpy as np
from matplotlib import pyplot as plt

from ann.genericAnn import ANN
from loss.losses import my_cross_entropy_loss
from normalization.normal import statisticalNormalisation, getStatisticalParameters, statisticalScalingParam
from utils.data import get_mnist
from utils.photoLoader import getImageMatrices


def normalize(images, params = []):
    feature_matrix = [[] for i in range(len(images[0]))]
    for image in images:
        for i in range(len(image)):
            feature_matrix[i].append(image[i])
    param_list = []
    normalized_matrix = []
    if len(params) == 0 :
        for i in range(len(feature_matrix)):
            mean,dev = getStatisticalParameters(feature_matrix[i])
            param_list.append([mean,dev])
            normalized_matrix.append(statisticalNormalisation(feature_matrix[i]))
    else:
        for i in range(len(feature_matrix)):
            mean = params[i][0]
            dev = params[i][1]
            normalized_matrix.append(statisticalScalingParam(feature_matrix[i], mean, dev))
        param_list = params

    new_images = [[] for i in range(len(images))]
    for i in range(len(normalized_matrix)):
        for j in range(len(images)):
            new_images[j].append(normalized_matrix[i][j])

    return new_images, param_list

def mnist():
    print("\nMNIST cod propriu (Poze 28x28 pixeli):")

    images, labels = get_mnist()
    np.random.seed(5)

    # folosesc doar 20% din setul de date
    # indexes = [i for i in range(len(images))]
    # sample = np.random.choice(indexes, int(0.01 * len(images)), replace=False)
    # half_images = [images[i] for i in sample]
    # images = half_images

    # print(images[0])
    # print(type(images))
    # print(type(images[0]))

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
    # trainOutputs = []
    # for l in trainLabels:
    #     trainOutputs.append(np.argmax(l))
    #
    # outputNames = [0,1,2,3,4,5,6,7,8,9]
    # plt.hist(trainOutputs, rwidth=0.8)
    # plt.xticks(np.arange(len(outputNames)), outputNames)
    # plt.show()

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

    over_fit_test = getImageMatrices("C:\Proiecte SSD\Python\LAB9AI\digitData\overfit_test", size=28)
    ovf = [np.concatenate(over_fit_test[i]) for i in range(len(over_fit_test))]
    for i in range(len(ovf)):
        img = ovf[i]
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        img.shape += (1,)
        p = ann.predict([img])
        p = np.concatenate(p, axis=0).flatten()
        # print(p.argmax())
        plt.xlabel("Predicted " + str(p.argmax()))
        plt.show()
        if p.argmax() == label.argmax():
            good += 1