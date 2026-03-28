import numpy as np
from Errors import InvalidForwardPass, InputError
from activation_functions import relu, leaky_relu, softmax, sigmoid, sigmoid_derivative
from loss_functions import binary_crossentropy, mean_squared_error, binary_crossentropy_prime
from data_processing import prep_data

import matplotlib.pyplot as plt

rng = np.random.default_rng()

class NeuralNetwork():
    def __init__(self, layers, network_shape, loss_metric):

        if layers != len(network_shape):
            raise ValueError("Network shape does not match layer count")

        self.layer_count = layers
        self.node_counts = {}

        self.weights = [[] for _ in range(self.layer_count)]
        self.biases = [[] for _ in range(self.layer_count)]
        self.values = [[] for _ in range(self.layer_count)]
        
        self.gradient_w = [None] * self.layer_count
        self.gradient_b = [None] * self.layer_count

        i = 0
        for node_count in network_shape:
            self.node_counts[i] = node_count
            if i > 0:
                # i should change the initial weight and biases range to see if changes make training faster/slower
                self.weights[i] = rng.random(size=(self.node_counts[i], self.node_counts[i - 1])) * 2 - 1
                self.biases[i] = rng.random((self.node_counts[i], 1)) * 2 - 1
            i += 1

        self.activation_functions = {
            "relu" : relu,
            "leaky_relu" : leaky_relu,
            "softmax" : softmax,
            "sigmoid" : [sigmoid, sigmoid_derivative]
        }

        self.loss_functions = {
            "binary_crossentropy" : [binary_crossentropy, binary_crossentropy_prime],
            "MSE" : mean_squared_error,
        }



    def input_values(self, input_array):

        if len(input_array) != self.node_counts[0]:
            raise InputError("Input array does not match size of network input size")
        
        self.values[0] = np.array(input_array)
    

    def forward_pass(self, layer, activation_function):

        if len(self.values[layer - 1]) == 0:
            raise InvalidForwardPass("Values for previous layer not yet calculated")
        
        self.values[layer] = np.dot(self.weights[layer], self.values[layer - 1]) + self.biases[layer]
        
        self.values[layer] = self.activation_functions[activation_function][0](self.values[layer])



    def cost(self, loss_metric, labels, batch_size):

        self.avg_cost = np.sum(self.loss_functions[loss_metric][0](self.values[self.layer_count - 1], labels)) / batch_size

        return -self.avg_cost
    
    def backward_pass(self):
        a = 0

    def backprop(self, loss_function, activation_function, labels, batch_size):
        self.propagate = self.loss_functions[loss_function][1](self.values[self.layer_count - 1], labels) / batch_size

        for layer in range(self.layer_count - 1, 0, -1):
            self.propagate = np.multiply(self.propagate, self.activation_functions[activation_function][1](self.values[layer]))

            weight_gradient = np.dot(self.propagate, self.values[layer - 1].T)
            bias_gradient = np.sum(self.propagate, axis=1, keepdims=True)

            self.gradient_w[layer] = weight_gradient
            self.gradient_b[layer] = bias_gradient

            self.propagate = np.dot(self.weights[layer].T, self.propagate)

        
    def update_params(self, learning_rate):
        for layer in range(1, self.layer_count):
            self.weights[layer] -= np.multiply(learning_rate, self.gradient_w[layer])
            self.biases[layer] -= np.multiply(learning_rate, self.gradient_b[layer])

    def calculate_accuracy(self, labels):
        a = 0
        




'''
NN.input_values(rng.random(size=(2, 500)))

NN.forward_pass(1, "sigmoid")
NN.forward_pass(2, "sigmoid")

NN.cost("binary_crossentropy", rng.integers(0, 2, size=((output_count, 500))), 500)

NN.backprop("binary_crossentropy", "sigmoid", rng.integers(0, 2, size=((output_count, 500))), 500)

print(NN.gradient_w)
print(NN.gradient_b)

'''

input_count = 30
output_count = 1

network_shape = [input_count, 256, 256, 256, 256, 256, 256, 256, output_count]
network_layers = len(network_shape)

NN = NeuralNetwork(network_layers, network_shape, 1)

x_inputs, y_labels = prep_data()
cost_history = []


def train(epochs):
    batch_size = 32
    samples = x_inputs.shape[1]
    print(samples)

    for epoch in range(epochs):

        permutation = np.random.permutation(samples)
        x_shuffled = x_inputs[:, permutation]
        y_shuffled = y_labels[:, permutation]

        for iteration in range(0, samples + 1, batch_size):

            x_batch = x_shuffled[:, iteration:iteration+batch_size]
            y_batch = y_shuffled[:, iteration:iteration+batch_size]

            current_batch_size = x_batch.shape[1] 

            NN.input_values(x_batch)

            for layer_pass in range(network_layers - 1):
                NN.forward_pass(layer_pass + 1, "sigmoid")

            avg_cost = NN.cost("binary_crossentropy", y_batch, current_batch_size)

            cost_history.append(avg_cost)


            print (f'Cost: {avg_cost}, epoch: {epoch} iteration: {iteration}')
            
            NN.backprop("binary_crossentropy", "sigmoid", y_batch, current_batch_size)
            NN.update_params(0.01)


train(10)

plt.figure(figsize=(10, 6))
plt.plot(cost_history, label="Cost")
plt.legend()
plt.grid(True)
plt.show()