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
def corrupter(input_image, corr_ratio):
    dim_row = input_image.shape[0]
    dim_col = input_image.shape[1]
    total_dim = dim_col*dim_row
    corrupted_image = cp.copy(input_image).flatten()

    points_to_corr = (total_dim*corr_ratio)/100
    for i in range(points_to_corr):
        corr_idx = rnd.randint(0, (dim_row*dim_col-1))
        corrupted_image[corr_idx] *= -1
    corrupted_image.shape = (dim_row, dim_col)

    return corrupted_image


# erase a part of the image starting from the top
def image_eraser(input_image, erase_ratio):
    dim_row = input_image.shape[0]
    dim_col = input_image.shape[1]
    erased_img = cp.copy(input_image).flatten()

    rows_to_erase = (dim_row*erase_ratio)/100
    for i in range(dim_col*rows_to_erase):
        erased_img[i] = 1

    erased_img.shape = (dim_row, dim_col)

    return erased_img


# Plot the results
def plotter(test_set, result_set):
    ntest = len(test_set)
    k = 1
    for i in range(ntest):
        plt.subplot(ntest, 2, k)
        plt.imshow(test_set[i], interpolation="nearest")
        k += 1
        plt.subplot(ntest, 2, k)
        plt.imshow(result_set[i], interpolation="nearest")
        k += 1
    plt.show()