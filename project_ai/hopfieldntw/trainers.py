__author__ = 'gbarbon'

import numpy as np


# train intput is in the form of a vector
def hebb_train(train_input, n_patterns, n_units):
    # weights matrix init to zeros
    weights = np.zeros((n_units, n_units))
    # weights = np.random.rand(n_units, n_units)

    # for i in range(n_units):
    #     for j in range(n_units):
    #         if i == j:
    #             continue
    #         # weights[i,j] = (1/n_units)* sum(...)
    #         for l in range(n_patterns):
    #             weights[i, j] += train_input[l, i] * train_input[l, j]
    #             # print("Temp weight at this point is", weights[i,j])
    #             # uguale???: weights[i,j] = sum(train_input[:,i]*train_input[:,j])

    for l in range(n_patterns):
        for i in range(n_units):
            for j in range(n_units):
                if i == j:
                    continue
            # weights[i,j] = (1/n_units)* sum(...)
                else:
                    weights[i, j] += train_input[l, i] * train_input[l, j]
                # print("Temp weight at this point is", weights[i,j])
                # uguale???: weights[i,j] = sum(train_input[:,i]*train_input[:,j])

    # for i in range(n_units):
    #     for j in range(n_units):
    #         if i == j:
    #             continue
    #         else:
    #             weights[i, j] = np.dot(train_input[:][i], train_input[:][j])

    weights *= 1 / float(n_units)
    return weights

    # nota: introdurre modifica: e' sufficiente calcolare solo la meta' superore della matrice!!

#  uses the storkey rule
def storkey_train(train_input, n_patterns, n_units):
    weights = np.zeros((n_units, n_units))

    #..#

    return weights

def q_pseudo_inv(train_input, n_patterns, n_units):
    q = np.zeros((n_patterns, n_patterns))

    for v in range(n_patterns):
        for u in range(n_patterns):
            # for i in range(n_units):
            #     q[u][v] += train_input[v][i]*train_input[u][i]
            q[u][v] = np.dot(train_input[v], train_input[u])

    q *= 1 / float(n_units)
    q = np.linalg.inv(q)

    return q

# uses the pseudo inverse training rule
def pseudo_inverse_train(train_input, n_patterns, n_units):
    weights = np.zeros((n_units, n_units))

    q = q_pseudo_inv(train_input, n_patterns, n_units)

    for v in range(n_patterns):
        for u in range(n_patterns):
            for i in range(n_units):
                for j in range(n_units):
                    if i == j:
                        continue
                    else:
                        weights[i, j] += train_input[v, i] * q[v][u] *train_input[u, j]


    weights *= 1 / float(n_units)
    return weights