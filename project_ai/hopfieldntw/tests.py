__author__ = 'gbarbon'

import numpy as np
import trainers as hb
import hopfield_net as net
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
weights = hb.hebb_train(train_input)

# crating threshold array (each unit has its own threshold)
threshold = np.zeros(weights.shape[0])

# creating test set
a_test = utl.corrupter(a_pattern, 5)
b_test = utl.corrupter(b_pattern, 5)
c_test = utl.corrupter(c_pattern, 5)

# testing the net
a_result = net.hopfield_net(weights, a_test, threshold)
b_result = net.hopfield_net(weights, b_test, threshold)
c_result = net.hopfield_net(weights, c_test, threshold)

#Show the results
test_set = np.array([a_test, b_test, c_test])
result_set = np.array([a_result, b_result, c_result])
utl.plotter(test_set, result_set)