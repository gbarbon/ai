__author__ = 'gbarbon'

import numpy as np
import HopfieldNet
import utils as utl
import imageManager as iM

# Config variables/constant
testnumber = 4
testel = 5  # elements to test
trainel = 10  # elements to train
corr_ratio = 0  # percentage of corruption ratio
erase_ratio = 0 # percentage of image erased
trainers = ["hebbian", "pseudoinv", "storkey"]
trainer = trainers[1]
filetype = "png"
all_trainer = False  # True for all trainers or False for only one
plotbool = True
savebool = True


def test1(trainer_type):
    # corruption_val = 5
    results_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/results/test_1"

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
    utl.plotter(test_set, result_set, results_dir, plotbool, savebool)


def test2(trainer_type, testel, trainel):
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/courier_digits_data_set/tiff_images_swidth"
    results_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/results/test_2" + "/" + trainer_type
    filename = results_dir + "/" + "tr" + str(trainel) + "_ts"+ str(testel)  + "_c" + str(corr_ratio) + "_e" + str(erase_ratio) + "_" + trainer_type + "." + filetype
    dim = [14, 9]  # in the form rows * cols
    # testel = 8  # elements for training
    #corruption_val = 5
    # trainers = ["hebbian","pseudoinv","storkey"]

    image_dim = [dim[1], dim[0]]  # changing shape for images

    # Loading images data set
    temp_train = iM.collectimages(image_dim, images_dir)

    # image conversion to 1 and -1 for Hopfield net
    for i in range(temp_train.shape[0]):
        temp = utl.image_converter(temp_train[i].reshape(dim))
        temp_train[i] = temp.flatten()

    train_input = np.zeros((trainel, dim[0] * dim[1]))
    for i in range(trainel):
        train_input[i] = temp_train[i]

    # training the net
    net = HopfieldNet.HopfieldNet(train_input, trainer_type, dim)

    # testing the net
    test_set = np.zeros((testel, dim[0], dim[1]))
    result_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        test_set[i] = temp_train[i].reshape(dim)
        if corr_ratio != 0:
            test_set[i] = utl.corrupter(test_set[i], corr_ratio)
        if erase_ratio != 0:
            test_set[i] = utl.image_eraser(test_set[i], erase_ratio)
        result_set[i] = net.test(test_set[i])

    # Plotting and saving results
    utl.plotter(test_set, result_set, filename, plotbool, savebool)


def test3(trainer_type, testel, trainel):
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/digital7_digit_data_set/tiff_images_rawcut"
    results_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/results/test_3" + "/" + trainer_type
    filename = results_dir + "/" + "tr" + str(trainel) + "_ts"+ str(testel)  + "_c" + str(corr_ratio) + "_e" + str(erase_ratio) + "_" + trainer_type + "." + filetype
    dim = [25, 16]  # in the form rows * cols
    # testel = 5  # elements for training
    #corruption_val = 10

    image_dim = [dim[1], dim[0]]  # changing shape for images

    # Loading images data set
    temp_train = iM.collectimages(image_dim, images_dir)

    # image conversion to 1 and -1 for Hopfield net
    for i in range(temp_train.shape[0]):
        temp = utl.image_converter(temp_train[i].reshape(dim))
        temp_train[i] = temp.flatten()

    train_input = np.zeros((trainel, dim[0] * dim[1]))
    for i in range(trainel):
        train_input[i] = temp_train[i]

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

    # Plotting and saving results
    utl.plotter(test_set, result_set, filename, plotbool, savebool)

def test_semeion(trainer_type, testel, trainel):
    images_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/dummy_data_set/courier_digits_data_set/tiff_images_swidth"
    results_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/results/test_semeion" + "/" + trainer_type
    semeion_dir = "/Users/jian/Dropbox/AI_dropbox/progetto_2014/semeion_data_set/semeion.data"
    filename = results_dir + "/" + "tr" + str(trainel) + "_ts"+ str(testel)  + "_c" + str(corr_ratio) + "_e" + str(erase_ratio) + "_" + trainer_type + "." + filetype
    dim = [16, 16]

    image_dim = [dim[1], dim[0]]  # changing shape for images

    # Loading images data set
    temp_train = iM.collectimages(image_dim, images_dir)

    # image conversion to 1 and -1 for Hopfield net
    for i in range(temp_train.shape[0]):
        temp = utl.image_converter(temp_train[i].reshape(dim))
        temp_train[i] = temp.flatten()

    train_input = np.zeros((trainel, dim[0] * dim[1]))
    for i in range(trainel):
        train_input[i] = temp_train[i]

    # training the net
    net = HopfieldNet.HopfieldNet(train_input, trainer_type, dim)

    # loading semeion data set
    test_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        test_set[i] = utl.semeion_loader(semeion_dir, i)

    # testing the net
    result_set = np.zeros((testel, dim[0], dim[1]))
    for i in range(testel):
        # test_set[i] = temp_train[i].reshape(dim)
        if corr_ratio != 0:
            test_set[i] = utl.corrupter(test_set[i], corr_ratio)
        if erase_ratio != 0:
            test_set[i] = utl.image_eraser(test_set[i], erase_ratio)
        result_set[i] = net.test(test_set[i])

    # Plotting and saving results
    utl.plotter(test_set, result_set, filename, plotbool, savebool)


    # loading semeion data set for training



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
            test2(iterator[i], testel, trainel)
        elif testnumber == 3:
            test3(iterator[i])
        elif testnumber == 4:
            test_semeion(iterator[i], testel, trainel)
    #total2()

def total2():
    test_couples = [[2,2],[2,3],[3,2],[5,5],[5,10],[8,8],[10,5],[10,10]]
    for i in range(len(trainers)):
        for j in range(len(test_couples)):
            testel = test_couples[j][1]
            trainel = test_couples[j][0]
            test2(trainers[i], testel, trainel)


if __name__ == "__main__":
    main()