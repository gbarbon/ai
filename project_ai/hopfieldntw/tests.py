__author__ = 'gbarbon'

import numpy as np
import HopfieldNet
import utils as utl
import imageManager as iM


def test1():
    # Create the training patterns
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
    net = HopfieldNet.HopfieldNet(train_input, "hebbian", [7, 7])

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
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/courier_digits_data_set/tiff_images_swidth"
    dim = [14, 9]  # in the form rows * cols
    testel = 8  # elements for training
    corruption_val = 30

    image_dim = [dim[1], dim[0]]  # changing shape for images

    # Loading images data set
    temp_train = iM.collectimages(image_dim, images_dir)

    train_input = np.zeros((testel, dim[0] * dim[1]))
    for i in range(testel):
        train_input[i] = temp_train[i]

    # image conversion to 1 and -1 for Hopfield net
    for i in range(train_input.shape[0]):
        temp = utl.image_converter(train_input[i].reshape(dim))
        train_input[i] = temp.flatten()

    # training the net
    net = HopfieldNet.HopfieldNet(train_input, "hebbian", dim)

    # testing the net
    test_set = np.zeros((testel, dim[0], dim[1]))
    result_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        test_set[i] = utl.corrupter(train_input[i].reshape(dim), corruption_val)
        result_set[i] = net.test(test_set[i])

    # Plotting results
    utl.plotter(test_set, result_set)

def test3():
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/digital7_digit_data_set/tiff_images_rawcut"
    dim = [25, 16]  # in the form rows * cols
    testel = 10  # elements for training
    corruption_val = 10

    image_dim = [dim[1], dim[0]]  # changing shape for images

    # Loading images data set
    temp_train = iM.collectimages(image_dim, images_dir)

    train_input = np.zeros((testel, dim[0] * dim[1]))
    for i in range(testel):
        train_input[i] = temp_train[i]

    # image conversion to 1 and -1 for Hopfield net
    for i in range(train_input.shape[0]):
        temp = utl.image_converter(train_input[i].reshape(dim))
        train_input[i] = temp.flatten()

    # training the net
    net = HopfieldNet.HopfieldNet(train_input, "pseudoinv", dim)

    # testing the net
    test_set = np.zeros((testel, dim[0], dim[1]))
    result_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        test_set[i] = utl.corrupter(train_input[i].reshape(dim), corruption_val)
        result_set[i] = net.test(test_set[i])

    # Plotting results
    utl.plotter(test_set, result_set)

test3()