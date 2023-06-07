import numpy as np

class NN:
    def __init__(self, epochs, batchSize, layers, neurons, activation, samples, labels, weights, bias):
        self.epochs = epochs
        self.batchSize = batchSize
        self.layers = layers
        self.neurons = neurons
        self.activation = activation
        self.input = samples
        self.labels = labels
        self.weights = weights
        self.bias = bias

        self.recall = 0
        self.precision = 0
        self.fitness = 0
        
    def test(self):
        pass

    def calculateFitness(self):
        if self.recall + self.precision > 0:
            F1_score = (self.precision * self.recall) / (self.precision + self.recall)
            if F1_score > 0.3:
                self.fitness = self.recall
            else:
                self.fitness = self.recall / 2
        else:
            self.fitness = self.recall

    def get_activation_function(self, func_name):
        if func_name == 'sigmoid':
            return self.sigmoid
        elif func_name == 'relu':
            return self.relu
        elif func_name == 'tanh':
            return self.tanh
        else:
            raise ValueError('Invalid activation function name')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def forward_propagation(self):
        layer_input = self.samples
        for i in range(self.layers):
            layer_output = np.dot(layer_input, self.weights[i]) + self.bias[i]
            activation_func = self.get_activation_function(self.activation[i])
            layer_input = activation_func(layer_output)
        return layer_input

    def evaluate(self):
        true_pred_counter = 0
        tp_counter = 0
        predicted_positive_counter = 0
        actual_positive_counter = 0

        output_probs = self.forward_propagation(self.samples)

        for i in range(len(self.samples)):
            y_hat = float(np.argmax(output_probs[i]))
            y_true = float(self.labels[i])

            if y_hat == y_true:
                true_pred_counter += 1

            if y_true == 1:
                actual_positive_counter += 1

            if y_hat == 1:
                predicted_positive_counter += 1
                if y_true == 1:
                    tp_counter += 1

        self.recall = tp_counter / actual_positive_counter
        self.precision = tp_counter / predicted_positive_counter

        self.calculateFitness()
