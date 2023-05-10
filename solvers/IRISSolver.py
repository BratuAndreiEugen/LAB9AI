import numpy as np
from matplotlib import pyplot as plt

from ann.genericAnn import ANN
from loss.losses import my_cross_entropy_loss
from solvers.MNISTSolver import normalize
from utils.data import get_mnist
from sklearn.datasets import load_iris, load_digits


def iris(): # [0,1,2] <=> [setosa, versicolor, virginica]
    print("IRIS cod propriu :\n")

    data = load_iris()
    outputs = []
    for op in data['target']:
        v = [0 for i in range(3)]
        v[op] = 1
        outputs.append(np.array(v))

    labels = np.array(outputs) # one hot
    inputs = data['data']


    # impartire test / train
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainLabels = [labels[i] for i in trainSample]
    validationInputs = [inputs[i] for i in validationSample]
    validationLabels = [labels[i] for i in validationSample]
    #print(validationLabels)

    # print(trainInputs[0])
    # normalizare ( nu e neaparat necesar -- sterge pana la spatiu )
    trainInputs, param = normalize(trainInputs)
    validationInputs, pr = normalize(validationInputs, param)

    t = []
    for img in trainInputs:
        t.append(np.array(img))
    trainInputs = t
    t = []
    for img in validationInputs:
        t.append(np.array(img))
    validationInputs = t

    trainOutputs = []
    for l in trainLabels:
        trainOutputs.append(np.argmax(l))

    outputNames = ['setosa', 'versicolor', 'virginica']
    plt.hist(trainOutputs, rwidth=0.8)
    plt.xticks(np.arange(len(outputNames)), outputNames)
    plt.show()

    # print(trainInputs[0])

    # antrenare
    ann = ANN(trainInputs, trainLabels, 3)  # inputs, outputs, neurons in hidden layer
    ann.train(learn_rate=0.1, activation=ANN.softmax, epochs=10)

    # testare

    # accuracy and cross entropy loss
    actual = []
    for label in validationLabels:
        actual.append(label.argmax())

    predicted = []
    good = 0
    for i in range(len(validationInputs)):
        img = validationInputs[i]
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

    print("Test Accuracy : " + str(good / len(validationInputs)))
    print("Cross-entropy loss : " + str(my_cross_entropy_loss(actual, predicted)))