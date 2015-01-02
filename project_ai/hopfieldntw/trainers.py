__author__ = 'gbarbon'

import numpy as np


# train intput is in the form of a vector
def hebb_train(train_input, n_patterns, n_units):
    # weights matrix init to zeros
    weights = np.zeros((n_units, n_units))
    # weights = np.random.rand(n_units, n_units)

    for i in range(n_units):
        for j in range(n_units):
            if i == j:
                continue
            # weights[i,j] = (1/n_units)* sum(...)
            for l in range(n_patterns):
                weights[i, j] += train_input[l, i] * train_input[l, j]
                # print("Temp weight at this point is", weights[i,j])
                # uguale???: weights[i,j] = sum(train_input[:,i]*train_input[:,j])

    weights *= 1 / float(n_units)
    return weights

    # nota: introdurre modifica: e' sufficiente calcolare solo la meta' superore della matrice!!