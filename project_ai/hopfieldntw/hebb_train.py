__author__ = 'jian'

import numpy as np

def hebb_train(train_input):

    # print("Input is")
    # print(train_input)

    #number of patterns
    n_patterns = train_input.shape[0]
    #print("Npatterns is", n_patterns)

    #number of units
    n_units = train_input.shape[1]
    #print("Nunits is", n_units)

    #weights matrix init to zeros
    weights = np.zeros((n_units, n_units))
    #print("Weights are")
    #print(weights)

    for i in range(n_units):
        for j in range(n_units):
            if i == j:
                continue
            #weights[i,j] = (1/n_units)* sum(...)
            for l in range(n_patterns):
                weights[i,j] += train_input[l,i]*train_input[l,j]
                #print("Temp weight at this point is", weights[i,j])
            #uguale???: weights[i,j] = sum(train_input[:,i]*train_input[:,j])

    # print("Before the end weights are")
    # print(weights)
    weights *= 1/float(n_units)

    # print("At the end weights are")
    # print(weights)

    return weights

#nota: e' sufficiente calcolare solo la meta' superore della matrice!!