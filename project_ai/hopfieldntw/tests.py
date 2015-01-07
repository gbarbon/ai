__author__ = 'gbarbon'

import numpy as np
import HopfieldNet
import utils as utl
import imageManager as iM

# Config variables/constant
testnumber = 2
testel = 3  # elements to test/train
corr_ratio = 10  # corruption ratio
erase_ratio = 20
trainers = ["hebbian", "pseudoinv", "storkey"]
# trainers = ["hebbian","pseudoinv","storkey","sanger"]
trainer = trainers[0]
all_trainer = False  # True for all trainers or False for only one


def test1(trainer_type):
    # corruption_val = 5

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
    net = HopfieldNet.HopfieldNet(train_input, trainer_type, [7, 7])

    # creating test set
    a_test = utl.corrupter(a_pattern, corr_ratio)
    b_test = utl.corrupter(b_pattern, corr_ratio)
    c_test = utl.corrupter(c_pattern, corr_ratio)

    # training and testing the net
    a_result = net.test(a_test)
    b_result = net.test(b_test)
    c_result = net.test(c_test)

    #Show the results
    test_set = np.array([a_test, b_test, c_test])
    result_set = np.array([a_result, b_result, c_result])
    utl.plotter(test_set, result_set)


def test2(trainer_type):
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/courier_digits_data_set/tiff_images_swidth"
    dim = [14, 9]  # in the form rows * cols
    # testel = 8  # elements for training
    #corruption_val = 5
    # trainers = ["hebbian","pseudoinv","storkey"]

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
    net = HopfieldNet.HopfieldNet(train_input, trainer_type, dim)

    # testing the net
    test_set = np.zeros((testel, dim[0], dim[1]))
    result_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        test_set[i] = train_input[i].reshape(dim)
        if corr_ratio != 0:
            test_set[i] = utl.corrupter(test_set[i], corr_ratio)
        if erase_ratio != 0:
            test_set[i] = utl.image_eraser(test_set[i], erase_ratio)
        result_set[i] = net.test(test_set[i])

    # Plotting results
    utl.plotter(test_set, result_set)


def test3(trainer_type):
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/digital7_digit_data_set/tiff_images_rawcut"
    dim = [25, 16]  # in the form rows * cols
    # testel = 5  # elements for training
    #corruption_val = 10

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
    net = HopfieldNet.HopfieldNet(train_input, trainer_type, dim)

    # testing the net
    test_set = np.zeros((testel, dim[0], dim[1]))
    result_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        test_set[i] = train_input[i].reshape(dim)
        if corr_ratio != 0:
            test_set[i] = utl.corrupter(test_set[i], corr_ratio)
        if erase_ratio != 0:
            test_set[i] = utl.image_eraser(test_set[i], erase_ratio)
        result_set[i] = net.test(test_set[i])

    # Plotting results
    utl.plotter(test_set, result_set)


def main():
    if all_trainer:
        iterator = trainers
    else:
        iterator = [trainer]
    for i in range(len(iterator)):
        print("Now trainer is: ", iterator[i])
        if testnumber == 1:
            test1(iterator[i])
        elif testnumber == 2:
            test2(iterator[i])
        elif testnumber == 3:
            test3(iterator[i])


if __name__ == "__main__":
    main()