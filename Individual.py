import numpy as np

class NN:
    def __init__(self,numOfLayers, layersSizes, activationFunctions, samples, lables):
        # self.layers = numOfLayers
        self.neurons = layersSizes
        self.activation = activationFunctions
        self.input = samples
        self.lables = lables
        

        self.recall = 0
        self.precision = 0
        self.fitness = 0
        
    def initializeWeights(self):
        self.weights = []
        # self.bias = []
        inputSize = len(self.input)
        for i in range(self.layers):
            self.weights.append(np.random.randn(inputSize, self.layersSizes[i]))
            # self.bias = 



    def test(self):
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(self.input)):
            a = self.input[i]
            for layer in range(len(self.weights)):
                z = np.add(np.dot(a, self.weights[layer]))
                a = sigmoid(z) 
            
            y_hat = float(np.argmax(softmax(a)))
            y_true = float(y[i])

    def calculateFitness(self):

        if self.recall+self.presicion > 0:
            F1_score = (self.presicion * self.recall) / \
                (self.presicion + self.recall)
            if F1_score > 0.3:
                self.fitness = self.recall
            else:
                self.fitness = self.recall/2
        else:
            self.fitness = self.recall
