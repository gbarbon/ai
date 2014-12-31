__author__ = 'gbarbon'

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
    #print(temp)
    if temp >= 0:
        # pattern[unit_idx] = 1
        return 1
    else:
        # pattern[unit_idx] = 0
        return -1

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

def hopfield_net(weights, ipattern, threshold):

    pattern = cp.copy(ipattern)
    prow = pattern.shape[0]
    pcols = pattern.shape[1]
    pattern = pattern.flatten()

    #energy init
    E = energy(weights, pattern, threshold)

    k = 0
    while k<10:
        randomRange = range(len(pattern))
        rd.shuffle(randomRange)
        for i in randomRange:
            pattern[i] = single_unit_updater(weights, i, pattern, threshold)
            #nota: l'energia deve essere calcolata ad ogni cambiamento di una singola unita' o di tutte le unita'?
        tempE = energy(weights, pattern, threshold)
        if tempE == E:
            #break while loop
            pattern.shape = (prow, pcols)
            return pattern
        else:
            E = tempE
        k +=1

    pattern.shape = (prow, pcols)
    return pattern