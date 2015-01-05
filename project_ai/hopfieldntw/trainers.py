__author__ = 'gbarbon'

import numpy as np
import time


# train intput is in the form of a vector
def hebb_train(train_input, n_patterns, n_units):
    # weights matrix init to zeros

    # # 1: original hebb algorithm
    # start = time.time()
    # w1 = np.zeros((n_units, n_units))
    # start = time.time()
    # for l in range(n_patterns):
    # for i in range(n_units):
    #         for j in range(i+1, n_units): # in order to compute only the upper half matrix
    #             w1[i, j] += train_input[l, i] * train_input[l, j]
    #             w1[j, i] = w1[i, j]
    # w1 *= 1 / float(n_units)
    # end = time.time()
    # elapsed = end - start
    # print("w1 elapsed time", elapsed)


    # # 2: hebb rule improved substituting the pattern loop with dot products
    # start = time.time()
    # w2 = np.zeros((n_units, n_units))
    # train_transp = zip(*train_input)
    # for i in range(n_units):
    #     for j in range(i+1, n_units): # in order to compute only the upper half matrix
    #         w2[i, j] = np.dot(train_transp[i], train_transp[j])
    #         w2[j, i] = w2[i, j]
    # w2 *= 1 / float(n_units)
    # end = time.time()
    # elapsed = end - start
    # print("w2 elapsed time", elapsed)


    # 3: hebb rule improved with matrix multiplication
    start = time.time()
    train_transp = zip(*train_input)  # matrix transpose
    w3 = np.dot(train_transp, train_input)
    w3 = w3.astype(float)
    w3 *= 1 / float(n_units)
    np.fill_diagonal(w3, 0)
    end = time.time()
    elapsed = end - start
    #print("w3 elapsed time", elapsed)
    print("Hebbian elapsed time", elapsed)

    # # Checking issues
    # if np.array_equal(w1,w2) and np.array_equal(w2, w3):
    #     print("All methods returns the same weight matrix.")
    # else:
    #     print("w1 and w2 ", np.array_equal(w1,w2))
    #     print("w2 and w3 ", np.array_equal(w2,w3))
    #     print("w3 and w1 ", np.array_equal(w1,w3))

    return w3


# support function fro pseudo inverse rule
def q_pseudo_inv(train_input, n_patterns, n_units):

    # # Original version with loop
    # start = time.time()
    # q1 = np.zeros((n_patterns, n_patterns))
    # for v in range(n_patterns):
    #     for u in range(n_patterns):
    #         for i in range(n_units):
    #             q1[u][v] += train_input[v][i] * train_input[u][i]
    # end = time.time()
    # elapsed = end - start
    # print("original version elapsed time", elapsed)

    # New version with dot product
    # start = time.time()
    q2 = np.zeros((n_patterns, n_patterns))
    for v in range(n_patterns):
        for u in range(n_patterns):
            q2[u][v] = np.dot(train_input[v], train_input[u])
    end = time.time()
    # elapsed = end - start
    # print("improved version elapsed time", elapsed)

    # # Checking issues
    # if np.array_equal(q1, q2):
    #     print("Both methods returns the same Q matrix.")

    q2 *= 1 / float(n_units)
    q = np.linalg.inv(q2)  # inverseof the matrix

    return q


# uses the pseudo inverse training rule
def pseudo_inverse_train(train_input, n_patterns, n_units):
    weights = np.zeros((n_units, n_units))

    # notice: the matrix is returned already inverse
    q = q_pseudo_inv(train_input, n_patterns, n_units)

    for v in range(n_patterns):
        for u in range(n_patterns):
            for i in range(n_units):
                for j in range(i + 1, n_units):  # in order to compute only the upper half matrix
                    weights[i, j] += train_input[v, i] * q[v][u] * train_input[u, j]
                    weights[j, i] = weights[i, j]

    weights *= 1 / float(n_units)
    return weights


# support function for storkey rule
def h_storkey(weights, i_index, j_index, pattern, pattern_idx, n_units):
    h = 0

    # start = time.time()
    for k in range(n_units):
        if k != i_index and k != j_index:
            h += weights[i_index][k] * pattern[pattern_idx][k]
    end = time.time()
    # elapsed1 = end - start

    # start = time.time()
    # h2 = 0
    # h2 = np.dot(weights[i_index], pattern[pattern_idx])
    # end = time.time()
    # h2 -= weights[i_index][i_index]*pattern[pattern_idx][i_index]
    # h2 -= weights[i_index][j_index]*pattern[pattern_idx][j_index]
    # # end = time.time()
    # elapsed2 = end - start
    #
    # if h==h2:
    # print("Results are the same")
    # else:
    #     print("RESULTS ARE DIFFERENT!! h1: ", h ," h2: ", h2)
    # print("First perf is ", elapsed1," while dot is ", elapsed2)

    return h


# uses the storkey rule
def storkey_train(train_input, n_patterns, n_units):
    weights = np.zeros((n_units, n_units))

    start = time.time()
    for l in range(n_patterns):
        for i in range(n_units):
            for j in range(i + 1, n_units):  # in order to compute only the upper half matrix
                temp = train_input[l, i] * train_input[l, j]
                temp -= train_input[l, i] * h_storkey(weights, j, i, train_input, l, n_units)
                temp -= h_storkey(weights, i, j, train_input, l, n_units) * train_input[l, j]
                temp *= 1 / float(n_units)
                weights[i, j] += temp
                weights[j, i] = weights[i, j]
        print(l, " iteration accomplished")
    end = time.time()
    elapsed = end - start
    print("Storkey algorithm execution time: ", elapsed)

    return weights