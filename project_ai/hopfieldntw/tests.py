__author__ = 'gbarbon'

import numpy as np
import HopfieldNet
import utils as utl
import imageManager as im

def test1():
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

def test2():
    dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/tiff_images_100x100"
    temp_train = im.collectimages([16, 16], dir)

    train_input = np.zeros((5, 16*16))
    for i in range(5):
        train_input[i] = temp_train[i]

    for i in range(train_input.shape[0]):
        temp = utl.image_converter(train_input[i].reshape(16,16))
        train_input[i] = temp.flatten()
        print(train_input[i].reshape(16,16))

    net = HopfieldNet.HopfieldNet(train_input, "hebbian", [16, 16])

    a_pattern = train_input[0].reshape(16, 16)
    b_pattern = train_input[1].reshape(16, 16)
    c_pattern = temp_train[2].reshape(16, 16)
    print(a_pattern)

    # creating test set
    a_test = utl.corrupter(a_pattern, 10)
    b_test = utl.corrupter(b_pattern, 40)
    c_test = utl.corrupter(c_pattern, 10)

    # dummy_array = np.zeros((16,16))
    # for i in range(dummy_array.shape[0]):
    #     for j in range(dummy_array.shape[1]):
    #         dummy_array[i][j] = -1
    # d_result = net.test(dummy_array)
    # dummy_2 = np.zeros((16, 16))
    # for i in range(dummy_2.shape[0]):
    #     for j in range(dummy_2.shape[1]):
    #         dummy_2[i][j] = 1
    # e_result = net.test(dummy_2)
    # print(dummy_array)
    # print(dummy_2)

    # training and testing the net
    a_result = net.test(a_test)
    b_result = net.test(b_test)
    c_result = net.test(c_test)

    #Show the results
    # test_set = np.array([a_pattern, dummy_2, dummy_array])
    # result_set = np.array([a_result, e_result, d_result])
    test_set = np.array([a_test, b_test, c_test])
    result_set = np.array([a_result, b_result, c_result])
    utl.plotter(test_set, result_set)

test2()