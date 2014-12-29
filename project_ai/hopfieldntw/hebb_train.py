__author__ = 'jian'

import numpy as np

def hebb_train(train_input):

    #number of patterns
    n_patterns = train_input.shape[0]

    #number of units
    n_units = train_input.shape[1]

    #weights matrix init to zeros
    weights = np.zeros((n_units, n_units))

    for i in range(n_units):
        for j in range(n_units):
            if i == j:
                continue
            #weights[i,j] = (1/n_units)* sum(...)
            for l in range(n_patterns):
                weights[i,j] += train_input[l,i]*train_input[l,j]
            #uguale???: weights[i,j] = sum(train_input[:,i]*train_input[:,j])
    weights *= 1/n_units

    return weights

#nota: e' sufficiente calcolare solo la meta' superore della matrice!!