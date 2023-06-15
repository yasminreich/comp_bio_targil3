import numpy as np
from copy import deepcopy

class NN:
    def __init__(self, inputSize, F1Threshold, numOfLayers, layersSizes, samples, lables, weights = None, fitness=0):
        self.numOfLayers = numOfLayers
        self.neurons = layersSizes
        # self.activation = activationFunctions
        self.input = samples
        self.lables = lables
        self.inputSize = inputSize
        self.F1Threshold = F1Threshold

        self.recall = 0
        self.precision = 0
        self.fitness = 0

        self.weights = weights
        if weights is None:
            self.initializeWeights()
        
        self.fitness = fitness
        # if fitness == None:
        #     self.calculateFitness()
        
    def initializeWeights(self):
        self.weights = []
        # self.bias = []
        inputSize = self.inputSize
        for i in range(self.numOfLayers):
            self.weights.append(np.random.randn(inputSize, self.neurons[i]))
            inputSize = self.neurons[i]
            # self.bias = 

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, z):
        return (np.exp(z)/np.exp(z).sum())


    def test(self):
        true_positive_counter = 0
        predicted_positive_counter = 0
        actual_positive_counter = 0
        for i in range(len(self.input)):
            a = self.input[i]
            for layer in range(self.numOfLayers):
                z = np.dot(a, self.weights[layer])
                a = self.sigmoid(z) 
            
            y_hat = float(np.argmax(self.softmax(a)))
            y_true = float(self.lables[i])

            if y_hat == 1:
                predicted_positive_counter += 1

            if y_true == 1:
                actual_positive_counter += 1

            if y_hat == 1 and y_true == 1:
                true_positive_counter += 1

        self.recall = true_positive_counter / actual_positive_counter
        if predicted_positive_counter == 0:
           self.precision = 0
        else: 
            self.precision = true_positive_counter / predicted_positive_counter


    def calculateFitness(self):

        if self.recall+self.precision > 0:
            F1_score = (self.precision * self.recall) / \
                (self.precision + self.recall)
            if F1_score > self.F1Threshold:
                self.fitness = self.recall
            else:
                self.fitness = self.recall/2
        else:
            self.fitness = self.recall

    
    def deepcopy(self):
        newPerson = NN(self.inputSize, self.F1Threshold, self.numOfLayers, self.layersSizes, self.samples, self.lables, deepcopy(self.weights), self.fitness)
        return newPerson
