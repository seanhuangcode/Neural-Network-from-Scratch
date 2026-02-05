import numpy as np

def sigmoid(np_array):
    return  1/(1+np.e**(-np_array))

def relu(np_array):
    return np.maximum(0, np_array)

def leaky_relu(np_array):
    return np.maximum(0.01 * np_array, np_array)

def softmax(np_array):

    for j in range(np_array.shape[1]):
        total = np.sum(np.exp(np_array[:, j]))

        np_array[:, j] = np.exp(np_array[:, j]) / total
    
    return np_array
