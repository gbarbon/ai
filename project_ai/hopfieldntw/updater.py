__author__ = 'jian'

import numpy as np
import random as rd
import copy as cp

def single_unit_updater(weights, unit_idx, pattern, threshold):

    # n_units = weights.shape[0]
    # temp = 0
    # for j in range(n_units):
    #     temp += weights[unit_idx,j]*pattern[j]
    # temp = temp - threshold[unit_idx]

    #temp = sum(weights[unit_idx,:]*pattern[:]) - threshold[unit_idx]
    # we implement a powerful version that uses dotproduct with numpy
    temp = np.dot(weights[unit_idx, :], pattern[:]) - threshold[unit_idx]
    print(temp)
    if temp >= 0:
        # pattern[unit_idx] = 1
        return 1
    else:
        # pattern[unit_idx] = 0
        return 0

def energy(weights, pattern, threshold):
    E = 0
    for i in range(len(pattern)):
        for j in range(len(pattern)):
            if i == j:
                continue
            else:
                E += weights[i][j]*pattern[i]*pattern[j]

    sub_term = np.dot(threshold, pattern)
    E = -1/2*E - sub_term

    return E

def updater(weights, ipattern, threshold):

    pattern = cp.copy(ipattern)
    #print("Initial pattern is")
    #print(pattern)
    print("Initial weights are")
    print(weights)

    #energy init
    E = energy(weights, pattern, threshold)
    print("The initial energy is ", E)

    k = 0
    while k<200:
        randomRange = range(len(pattern))
        rd.shuffle(randomRange)
        for i in randomRange:
            pattern[i] = single_unit_updater(weights, i, pattern, threshold)
            tempE = energy(weights, pattern, threshold)
            #print("Temp energy at iteration ", i, " is ", E)
            if tempE == E:
                #break
                return pattern
            else:
                E = tempE
        k +=1

    return pattern
