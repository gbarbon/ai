__author__ = 'gbarbon'

import numpy as np
import random as rd
import trainers as tr


class HopfieldNet:

    def __init__(self, train_input, trainer_type, image_dimensions):

        #number of training patterns and number of units
        n_patterns = train_input.shape[0]
        self.n_units = train_input.shape[1]

        # crating threshold array (each unit has its own threshold)
        self.threshold = np.zeros(self.n_units)

        # setting image dimension
        self.image_dimensions = image_dimensions

        # net training
        if trainer_type == "hebbian":
            self.weights = tr.hebb_train(train_input, n_patterns, self.n_units)
        elif trainer_type == "pseudoinv":
            self.weights = tr.pseudo_inverse_train(train_input, n_patterns, self.n_units)
        elif trainer_type == "storkey":
            self.weights = tr.storkey_train(train_input, n_patterns, self.n_units)
        #else:

    def single_unit_updater(self, unit_idx, pattern):

        #temp = sum(weights[unit_idx,:]*pattern[:]) - threshold[unit_idx]
        # we implement a powerful version that uses dotproduct with numpy
        temp = np.dot(self.weights[unit_idx, :], pattern[:]) - self.threshold[unit_idx]
        if temp >= 0:
            # pattern[unit_idx] = 1
            return 1
        else:
            # pattern[unit_idx] = 0
            return -1

    def energy(self, pattern):
        e = 0
        length = len(pattern)
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                else:
                    e += self.weights[i][j] * pattern[i] * pattern[j]

        sub_term = np.dot(self.threshold, pattern)
        e = -1 / 2 * e - sub_term
        return e

    # function overloading in threshold
    def test(self, pattern, threshold=0):

        #setting threshold if threshold != 0
        if threshold != 0:
            self.threshold = threshold

        pattern = pattern.flatten()  # flattening pattern
        energy = self.energy(pattern)  # energy init

        k = 0
        while k < 10:
            randomrange = range(len(pattern))
            rd.shuffle(randomrange)
            for i in randomrange:
                pattern[i] = self.single_unit_updater(i, pattern)
                #nota: l'energia deve essere calcolata ad ogni cambiamento di una singola unita' o di tutte le unita'?
            temp_e = self.energy(pattern)
            if temp_e == energy:
                #break while loop
                pattern.shape = (self.image_dimensions[0], self.image_dimensions[1])
                return pattern
            else:
                energy = temp_e
            k += 1

        pattern.shape = (self.image_dimensions[0], self.image_dimensions[1])
        return pattern