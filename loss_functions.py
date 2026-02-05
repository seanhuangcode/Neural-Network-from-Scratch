import numpy as np

epsilon = 1e-15

def binary_crossentropy(np_array, np_labels):

    np_array = np.clip(np_array, epsilon, 1 - epsilon)
    np_labels = np.clip(np_labels, epsilon, 1 - epsilon)

    for j in range(len(np_array[0])):
        np_array[:, j] = np.multiply(np_labels[:, j], np.log(np_array[:, j])) + \
        np.multiply(1 - np_labels[:, j], np.log(1 - np_array[:, j]))

    return np_array

def mean_squared_error(np_array, np_labels):

    costs = np.array([])

    np_losses = np.square(np_labels - np_array)

    for j in range(np_array.shape[1]):
        costs = np.append(costs, np.sum(np_losses[:, j]) / np_array.shape[0])

    costs = costs.reshape(1, 500)


    
    return costs
        




