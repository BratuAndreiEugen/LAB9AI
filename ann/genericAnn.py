import numpy as np


class ANN:
    def __init__(self, inputs, outputs, hidden):
        self.inputs = inputs # [[feature1, feature2, ..],..]
        self.outputs = outputs # one hot
        self.hidden = hidden
        self.w_input_hidden = np.random.uniform(-0.5, 0.5, (hidden, len(inputs[0])))
        self.w_hidden_output = np.random.uniform(-0.5, 0.5, (len(outputs[0]), hidden))
        self.b_input_hidden = np.zeros((hidden, 1))
        self.b_hidden_output = np.zeros((len(outputs[0]), 1))
        self.activation = None

    def train(self, activation, learn_rate = 0.01, epochs = 3):
        self.activation = activation
        corect = 0
        for epoch in range(epochs):
            for input, output in zip(self.inputs, self.outputs):
                if epoch == 0:
                    input.shape += (1,)
                    output.shape += (1,)

                # forward propagation input->hidden
                hidden_predict = self.b_input_hidden + self.w_input_hidden @ input
                h = self.sigmoid(hidden_predict) # activation

                # forward propagation hidden->output
                output_predict = self.b_hidden_output + self.w_hidden_output @ h
                o = activation(self, output_predict) # activation

                # error
                # print("o : ", o)
                # print("op : ", output)
                e = 1 / len(o) * np.sum((o - output)**2, axis=0)
                corect += int(np.argmax(o) == np.argmax(output))

                # back propagation output->hidden
                delta_o = o - output
                #print("delta_o : ", delta_o)
                self.w_hidden_output += -learn_rate * delta_o @ np.transpose(h)
                self.b_hidden_output += -learn_rate * delta_o

                # back propagation hidden->input
                delta_h = np.transpose(self.w_hidden_output) @ delta_o * (h * (1 - h))
                self.w_input_hidden += -learn_rate * delta_h @ np.transpose(input)
                self.b_input_hidden += -learn_rate * delta_h

            # accuracy for epoch
            print(f"Accuracy: {round((corect / len(self.inputs)) * 100, 2)}%")
            corect = 0

    def predict(self, validationInputs):
        op = []
        for input in validationInputs:
            input.shape += (1,)
            # forward propagation input -> hidden
            h_pre = self.b_input_hidden + self.w_input_hidden @ input.reshape(len(self.inputs[0]), 1)
            h = self.sigmoid(h_pre)
            # forward prpg hidden -> output
            o_pre = self.b_hidden_output + self.w_hidden_output @ h
            o = self.activation(self, o_pre)
            op.append(o)
            # print(o)
            # print(sum(o))

        return op

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

