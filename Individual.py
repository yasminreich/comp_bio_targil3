import numpy as np
from copy import deepcopy
import pickle


class NN:
    def __init__(self, inputSize, F1Threshold, numOfLayers, layersSizes, samples, lables, weights=None, recall=0, precision=0, accuray=0):
        self.numOfLayers = numOfLayers
        self.neurons = layersSizes
        # self.activation = activationFunctions
        self.input = samples
        self.lables = lables
        self.inputSize = inputSize
        self.F1Threshold = F1Threshold

        self.recall = recall
        self.precision = precision

        self.accuracy = accuray

        self.weights = weights
        if weights is None:
            self.initializeWeights()


    def initializeWeights(self):
        self.weights = []
        self.bias = []
        inputSize = self.inputSize
        for i in range(self.numOfLayers):
            self.weights.append(np.random.randn(inputSize, self.neurons[i]))
            inputSize = self.neurons[i]

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, z):
        return (np.exp(z)/np.exp(z).sum())

    # def train(self):
    #     y_hat = []
    #     for i in range(self.input.shape[0]):
    #         a = self.input[i].reshape(1,self.inputSize)
    #         for layer in range(self.numOfLayers):
    #             z = np.dot(a, self.weights[layer])
    #             a = self.sigmoid(z)
    #         y_hat.append(float(np.argmax(self.softmax(a))))

    #     self.calculate_metrics(np.array(y_hat))


    def train(self):
        a = self.input.reshape(self.input.shape[0], 1, self.inputSize)
        for layer in range(self.numOfLayers):
            z = np.dot(a, self.weights[layer])
            a = self.sigmoid(z)
        
        # Flatten the output array
        a = a.reshape(a.shape[0], -1)
        
        # Calculate y_hat for all samples
        y_hat = np.argmax(self.softmax(a), axis=1)
        
        self.accuracy = np.mean(y_hat == self.lables)

    def test(self, samples, labels):
        true_pred_counter = 0
        for i in range(len(samples)):
            a = samples[i]
            for layer in range(self.numOfLayers):
                z = np.dot(a, self.weights[layer])
                a = self.sigmoid(z)

            y_hat = float(np.argmax(self.softmax(a)))
            y_true = float(labels[i])

            if y_hat == y_true:
                true_pred_counter += 1

        accuracy = true_pred_counter / len(labels)
        return accuracy

    def deepcopy(self):
        newPerson = NN(self.inputSize, self.F1Threshold, self.numOfLayers, self.neurons, self.input, self.lables,
                       deepcopy(self.weights), self.recall, self.precision, self.accuracy)
        return newPerson

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)


