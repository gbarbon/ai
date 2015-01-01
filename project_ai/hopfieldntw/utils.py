__author__ = 'gbarbon'

from matplotlib import pyplot as plt
import random as rnd
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
        corr_idx = rnd.randint(0, (dim_row*dim_col-1))
        corrupted_image[corr_idx] *= -1
    corrupted_image.shape = (dim_row, dim_col)

    return corrupted_image


# Plot the results
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