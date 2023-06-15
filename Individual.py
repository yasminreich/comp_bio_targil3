import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score


class NN:
    def __init__(self, inputSize, F1Threshold, numOfLayers, layersSizes, samples, lables, weights=None, fitness=0, recall=0, precision=0):
        self.numOfLayers = numOfLayers
        self.neurons = layersSizes
        # self.activation = activationFunctions
        self.input = samples
        self.lables = lables
        self.inputSize = inputSize
        self.F1Threshold = F1Threshold

        self.recall = recall
        self.precision = precision

        self.fitness = 0

        self.weights = weights
        if weights is None:
            self.initializeWeights()

        self.fitness = fitness
        # if fitness == None:
        #     self.calculateFitness()

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

    def train(self):
        y_hat = []
        for i in range(self.input.shape[0]):
            a = self.input[i].reshape(1,self.inputSize)
            for layer in range(self.numOfLayers):
                z = np.dot(a, self.weights[layer])
                a = self.sigmoid(z)
            y_hat.append(float(np.argmax(self.softmax(a))))

        # y_hat = np.round(a.transpose())

        self.calculate_metrics(np.array(y_hat))

    def calculate_metrics(self, predictions):

        # Calculate accuracy
        self.accuracy = np.mean(predictions == self.lables)

        # Calculate true positives, true negatives, false positives, and false negatives
        true_positives = np.sum(np.logical_and(predictions == 1, self.lables == 1))
        true_negatives = np.sum(np.logical_and(predictions == 0, self.lables == 0))
        false_positives = np.sum(np.logical_and(predictions == 1, self.lables == 0))
        false_negatives = np.sum(np.logical_and(predictions == 0, self.lables == 1))

        # Calculate precision
        self.precision = true_positives / (true_positives + false_positives)

        # Calculate recall
        self.recall = true_positives / (true_positives + false_negatives)

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

    def calculateFitness(self):
        # self.fitness = self.accuracy
        if self.recall+self.precision > 0:
            F1_score = (self.precision * self.recall) / \
                (self.precision + self.recall)
            if F1_score > self.F1Threshold:
                self.fitness = F1_score
            else:
                self.fitness = self.recall/2
        else:
            self.fitness = self.recall

    def deepcopy(self):
        newPerson = NN(self.inputSize, self.F1Threshold, self.numOfLayers, self.neurons, self.input, self.lables,
                       deepcopy(self.weights), self.fitness, self.recall, self.precision)
        return newPerson
