__author__ = 'jian'

from matplotlib import pyplot as plt

import random as rnd
import numpy as np
import hebb_train as hb
import updater as net


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

train_input = np.array([a_pattern.flatten(), b_pattern.flatten(), c_pattern.flatten()])
weights = hb.hebb_train(train_input)
threshold = np.zeros(weights.shape[0])

a_test =  a_pattern.flatten()
for i in range(20):
    p = rnd.randint(0, 48)
    a_test[p] *= 0

a_result = net.updater(weights, a_test, threshold)

a_result.shape = (7, 7)
a_test.shape = (7, 7)

b_test =  b_pattern.flatten()
for i in range(2):
    p = rnd.randint(0, 48)
    b_test[p] *= 0

b_result = net.updater(weights, b_test, threshold)

b_result.shape = (7, 7)
b_test.shape = (7, 7)

c_test =  c_pattern.flatten()
for i in range(5):
    p = rnd.randint(0, 48)
    c_test[p] *= 0

c_result = net.updater(weights, c_test, threshold)

b_result.shape = (7, 7)
b_test.shape = (7, 7)

#Show the results
# plt.subplot(3, 2, 1)
# plt.imshow(a_test, interpolation="nearest")
# plt.subplot(3, 2, 2)
# plt.imshow(a_result, interpolation="nearest")
#
# plt.subplot(3, 2, 3)
# plt.imshow(b_test, interpolation="nearest")
# plt.subplot(3, 2, 4)
# plt.imshow(b_result, interpolation="nearest")

# plt.subplot(3, 2, 5)
# plt.imshow(c_test, interpolation="nearest")
# plt.subplot(3, 2, 6)
# plt.imshow(c_result, interpolation="nearest")

#plt.show()