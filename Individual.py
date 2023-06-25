import numpy as np
from copy import deepcopy
import pickle

#class for a neural network optional selution
class NN:
    def __init__(self, inputSize, numOfLayers, layersSizes, samples, lables, weights=None, recall=0, precision=0, accuray=0):
        self.numOfLayers = numOfLayers
        self.neurons = layersSizes
        self.input = samples
        self.lables = lables
        self.inputSize = inputSize

        self.recall = recall
        self.precision = precision

        self.accuracy = accuray

        self.weights = weights
        if weights is None:
            self.initializeWeights()

    #the weights are initialized randomly
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

    #run the neural network on the train set and returns the accuracy
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

    #run the network on the test set samples and returns the labels
    def test(self, samples):

        a = samples.reshape(samples.shape[0], 1, self.inputSize)
        for layer in range(self.numOfLayers):
            z = np.dot(a, self.weights[layer])
            a = self.sigmoid(z)

        # Flatten the output array
        a = a.reshape(a.shape[0], -1)
        
        # Calculate y_hat for all samples
        y_hat = np.argmax(self.softmax(a), axis=1)
        return y_hat

    def deepcopy(self):
        newPerson = NN(self.inputSize, self.numOfLayers, self.neurons, self.input, self.lables,
                       deepcopy(self.weights), self.recall, self.precision, self.accuracy)
        return newPerson

    #for saving the final network to the wnet file
    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)


