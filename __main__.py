import numpy as np
from Errors import InvalidForwardPass, InputError
from activation_functions import relu, leaky_relu, softmax, sigmoid

rng = np.random.default_rng()

class NeuralNetwork():
    def __init__(self, layers, network_shape, loss_metric, learning_rate, samples):

        if layers != len(network_shape):
            raise ValueError("Network shape does not match layer count")

        self.layer_count = layers
        self.node_counts = {}

        self.weights = [[] for _ in range(self.layer_count)]
        self.biases = [[] for _ in range(self.layer_count)]
        self.values = [[] for _ in range(self.layer_count)]

        self.samples = samples

        self.activation_functions = {
            "relu" : relu,
            "leaky_relu" : leaky_relu,
            "softmax" : softmax,
            "sigmoid" : sigmoid,
        }

        self.inputed = False


        i = 0
        for node_count in network_shape:
            self.node_counts[i] = node_count
            if i > 0:
                # i should change the initial weight and biases range to see if changes make training faster/slower
                self.weights[i] = rng.random(size=(self.node_counts[i], self.node_counts[i - 1])) * 2 - 1
                self.biases[i] = np.array([rng.random(size=[self.node_counts[i]]) * 2 - 1])
            i += 1

    def input_values(self, input_array):
        if self.inputed:
            raise InputError("Values already inputted")

        if len(input_array) != self.node_counts[0]:
            raise InputError("Input array does not match size of network input size")
        
        self.values[0] = np.array(input_array)
    

    def forward_pass(self, layer, activation_function):

        if len(self.values[layer]) != 0:
            raise InvalidForwardPass("Values for this layer already calculated")

        if len(self.values[layer - 1]) == 0:
            raise InvalidForwardPass("Values for previous layer  not yet calculated")
        
        self.values[layer] = np.dot(self.weights[layer], self.values[layer - 1]) + self.biases[layer].T
        
        self.values[layer] = self.activation_functions[activation_function](self.values[layer])

NN = NeuralNetwork(4, [10, 100, 100, 2], 1, 1, 500)

NN.input_values(rng.random(size=(10, 500)))
NN.forward_pass(1, "sigmoid")
NN.forward_pass(2, "sigmoid")
NN.forward_pass(3, "softmax")

print (NN.values[3])