__author__ = 'jian'

from matplotlib import pyplot as plt

import random as rnd
import numpy as np
import hebb_train as hb
import updater as net
import copy as cp

# convert images from 0/1 to -1/1
def image_converter(input_image):
    image = cp.copy(input_image)
    image *= 2
    image -= 1
    return image

# corrupts images
def corrupter(input_image, corrupt_param):
    dim_row = input_image.shape[0]
    dim_col = input_image.shape[1]
    corrupted_image = cp.copy(input_image).flatten()

    for i in range(corrupt_param):
        corr_idx = rnd.randint(0, 48)
        corrupted_image[corr_idx] *= -1
    corrupted_image.shape = (dim_row,dim_col)

    return corrupted_image

#Plot the results
def plotter(test_set, result_set):
    k = 1
    for i in range(len(test_set)):
        plt.subplot(3, 2, k)
        plt.imshow(test_set[i], interpolation="nearest")
        k += 1
        plt.subplot(3, 2, k)
        plt.imshow(result_set[i], interpolation="nearest")
        k += 1
    plt.show()

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

a_pattern = image_converter(a_pattern)
b_pattern = image_converter(b_pattern)
c_pattern = image_converter(c_pattern)

train_input = np.array([a_pattern.flatten(), b_pattern.flatten(), c_pattern.flatten()])
#train_input = np.array([b_pattern.flatten(), c_pattern.flatten()])

#hebbian training
weights = hb.hebb_train(train_input)

# crating threshold array (each unit has its own threshold)
threshold = np.zeros(weights.shape[0])

# creating test set
a_test = corrupter(a_pattern, 5)
b_test = corrupter(b_pattern, 5)
c_test = corrupter(c_pattern, 5)

# testing the net
a_result = net.updater(weights, a_test.flatten(), threshold)
b_result = net.updater(weights, b_test.flatten(), threshold)
c_result = net.updater(weights, c_test.flatten(), threshold)

# pattern in the matrix shape
a_result.shape = (7, 7)
a_test.shape = (7, 7)
b_result.shape = (7, 7)
b_test.shape = (7, 7)
c_result.shape = (7, 7)
c_test.shape = (7, 7)

#Show the results
test_set = np.array([a_test, b_test, c_test])
result_set = np.array([a_result, b_result, c_result])

plotter(test_set, result_set)