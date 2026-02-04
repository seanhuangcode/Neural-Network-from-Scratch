import numpy as np

rng = np.random.default_rng()

class NeuralNetwork():
    def __init__(self, layers, network_shape, activation_function, loss_metric, learning_rate):

        if layers != len(network_shape):
            raise ValueError("Network shape does not match layer count")

        self.layer_count = layers
        self.node_counts = {}
        self.weights = {}
        self.biases = {}

        i = 0
        for node_count in network_shape:
            self.node_counts[i] = node_count
            if i > 0:
                # i should change the initial weight and biases range to see if changes make training faster/slower
                self.weights[i] = rng.random(size=(self.node_counts[i], self.node_counts[i-1])) * 2 - 1
                self.biases[i] = rng.random(size=[self.node_counts[i]]) * 2 -1

            i += 1
        
        


NeuralNetwork(4, [10, 100, 100, 10], 1, 1, 1)