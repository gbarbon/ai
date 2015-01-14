__author__ = 'gbarbon'

from matplotlib import pyplot as plt
import numpy as np
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

    points_to_corr = int((total_dim*corr_ratio)/100)
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

    rows_to_erase = int((dim_row*erase_ratio)/100)
    for i in range(dim_col*rows_to_erase):
        erased_img[i] = 1

    erased_img.shape = (dim_row, dim_col)

    return erased_img

#
def semeion_loader(semeion_dir, el):
    data = np.loadtxt(semeion_dir)
    n_el = data.shape[0]
    found = False
    i = rnd.randint(0, n_el)
    while ( not found):
        if data[i][256+el] == 1:
            found = True
            image = cp.copy(data[i])
        else:
            i+=1
        if i>=n_el:
            i = 0
    image = np.delete(image, [256,257,258,259,260,261,262,263,264,265,266]) # remove semeion label
    image = image_converter(image)
    image = image.reshape(16,16)
    return image


# Plot and/or save the results
def plotter(test_set, result_set, filename, plotbool, savebool):
    ntest = len(test_set)
    tickslabels_array_big = ([-0.5, 2.5, 5.5, 8.5], ['0', '3', '6', '9'], [-0.5, 2.5, 5.5, 8.5, 11.5], ['0', '3', '6', '9', '12'])
    tickslabels_array_small = ([-0.5, 4.5, 8.5], ['0', '5', '9'], [-0.5, 5.5, 11.5], ['0', '6', '12'])

    k = 1
    if (ntest>5):
        tickslabels_array = tickslabels_array_small
        lsize = 6
    else:
        tickslabels_array = tickslabels_array_big
        lsize = 8
    fig=plt.figure()
    for i in range(ntest):

        tmp = fig.add_subplot(ntest, 2, k)
        tmp.imshow(test_set[i], "summer",interpolation="nearest")

        tmp.tick_params(labelsize=lsize)
        tmp.set_xticks(tickslabels_array[0])
        tmp.set_xticklabels(tickslabels_array[1])
        tmp.set_yticks(tickslabels_array[2])
        tmp.set_yticklabels(tickslabels_array[3])
        if k==1:
            #tmp.set_title("Test set")
            tmp.text(.5, 1.2, 'Test set', horizontalalignment='center', transform=tmp.transAxes, fontsize=16)
        k += 1

        tmp = fig.add_subplot(ntest, 2, k)
        tmp.imshow(result_set[i], "winter", interpolation="nearest")
        tmp.tick_params(labelsize=lsize)
        tmp.set_xticks(tickslabels_array[0])
        tmp.set_xticklabels(tickslabels_array[1])
        tmp.set_yticks(tickslabels_array[2])
        tmp.set_yticklabels(tickslabels_array[3])
        if k==2:
            tmp.text(.5, 1.2, 'Results', horizontalalignment='center', transform=tmp.transAxes, fontsize=16)
        k += 1
    fig.subplots_adjust(hspace=.7, wspace=0.01)
    if plotbool:
        plt.show()
    if savebool:
        fig.savefig(filename, bbox_inches='tight')
