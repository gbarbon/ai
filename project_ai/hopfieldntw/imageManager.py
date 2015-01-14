__author__ = 'gbarbon'

from PIL import Image
from PIL import ImageFilter
# from PIL import ImageEnhance
import os
import numpy as np


# NOTE: in images dimension are: first:= columns, second:= rows

def image_cropper(image, new_dimensions):
    width, height = image.size  # Get dimensions

    left = (width - new_dimensions[1]) / 2
    top = (height - new_dimensions[0]) / 2
    right = (width + new_dimensions[1]) / 2
    bottom = (height + new_dimensions[0]) / 2

    return image.crop((left, top, right, bottom))


def image_resizer(image, new_dimensions):
    return image.resize(new_dimensions)


def to_greyscale(image):
    return image.convert("L")


def to_blackwhite(input_image):
    return input_image.convert("1")


# convert image to a matrix of 0 and 1 and to black and white
def tomatrix_bew(image):
    imarray = np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0])
    row = imarray.shape[0]
    cols = imarray.shape[1]
    newarray = np.zeros((row, cols))

    for i in range(row):
        for j in range(cols):
            if imarray[i][j] <= 127:
                newarray[i][j] = 1
            else:
                newarray[i][j] = 0
    return newarray


# convert image to a matrix of 0 and 1
def tomatrix(image):
    matrix = np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0])
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    newarray = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                newarray[i][j] = 1

    return newarray


def collectimages(finaldim, img_dir, filter):
    i = 0
    entries = 0

    # number of files checking
    for img_file in os.listdir(img_dir):
        if img_file.endswith(".tiff"):
            entries += 1

    dataset = np.zeros((entries, finaldim[0] * finaldim[1]))

    for img_file in os.listdir(img_dir):
        if img_file.endswith(".tiff"):
            newdir = img_dir + "/" + img_file
            im = Image.open(newdir)

            orig_dim = im.size

            # Image conversion to black and white
            imp = to_greyscale(im)
            imp = to_blackwhite(imp)

            # Image filtering
            if filter == "median":
                imp = imp.filter(ImageFilter.MedianFilter(size=5))


            if orig_dim[0] > 100 and orig_dim[1] > 100:
                # crop if dimensions higher than 100
                imp = image_cropper(imp, [100, 100])
                imp = image_resizer(imp, finaldim)
            else:
                imp = image_resizer(imp, [40, 70])
                imp = image_resizer(imp, finaldim)

            #imp.show()  # shows image in external program
            imarray = tomatrix(imp)
            dataset[i] = imarray.flatten()
            i += 1

    return dataset