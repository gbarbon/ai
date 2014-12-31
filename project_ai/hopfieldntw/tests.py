__author__ = 'gbarbon'

import numpy as np
import HopfieldNet
import utils as utl

#Create the training patterns
a_pattern = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 1, 1, 1, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0]])

b_pattern = np.array([[0, 1, 1, 1, 1, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 1, 1, 1, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 1, 1, 1, 1, 0, 0]])

c_pattern = np.array([[0, 1, 1, 1, 1, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 0]])

a_pattern = utl.image_converter(a_pattern)
b_pattern = utl.image_converter(b_pattern)
c_pattern = utl.image_converter(c_pattern)

train_input = np.array([a_pattern.flatten(), b_pattern.flatten(), c_pattern.flatten()])

#hebbian training
net = HopfieldNet.HopfieldNet(train_input, "hebbian", [7,7])

# creating test set
a_test = utl.corrupter(a_pattern, 5)
b_test = utl.corrupter(b_pattern, 5)
c_test = utl.corrupter(c_pattern, 5)

# training and testing the net
a_result = net.test(a_test)
b_result = net.test(b_test)
c_result = net.test(c_test)

#Show the results
test_set = np.array([a_test, b_test, c_test])
result_set = np.array([a_result, b_result, c_result])
utl.plotter(test_set, result_set)