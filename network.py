import numpy as np
from Errors import InvalidForwardPass, InputError
from activation_functions import relu, leaky_relu, softmax, sigmoid, sigmoid_derivative
from loss_functions import binary_crossentropy, mean_squared_error, binary_crossentropy_prime

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




'''
NN.input_values(rng.random(size=(2, 500)))

NN.forward_pass(1, "sigmoid")
NN.forward_pass(2, "sigmoid")

NN.cost("binary_crossentropy", rng.integers(0, 2, size=((output_count, 500))), 500)

NN.backprop("binary_crossentropy", "sigmoid", rng.integers(0, 2, size=((output_count, 500))), 500)

print(NN.gradient_w)
print(NN.gradient_b)

'''

input_count = 2
output_count = 1

NN = NeuralNetwork(5, [input_count, 100, 100, 100, output_count], 1)

y_labels = rng.integers(0, 2, size=(output_count, 50))
x_inputs = rng.random(size=(2, 50))

def train(epochs):
    batch_size = 50
    for epoch in range(epochs):
        for iteration in range(0, 1001, batch_size):
            NN.input_values(x_inputs)

            NN.forward_pass(1, "sigmoid")
            NN.forward_pass(2, "sigmoid")
            NN.forward_pass(3, "sigmoid")
            NN.forward_pass(4, "sigmoid")

            avg_cost = NN.cost("binary_crossentropy", y_labels, batch_size)

            print (f'Cost: {avg_cost}, epoch: {epoch} iteration: {iteration}')
            
            NN.backprop("binary_crossentropy", "sigmoid", y_labels, batch_size)
            NN.update_params(0.1)

train(2000)

