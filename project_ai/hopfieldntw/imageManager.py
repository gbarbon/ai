__author__ = 'gbarbon'

from PIL import Image
import os
import numpy as np

# NOTE: in images dimension are: first:= columns, second:= rows

def image_cropper(image, new_dimensions):
    width, height = image.size   # Get dimensions

    left = (width - new_dimensions[1])/2
    top = (height - new_dimensions[0])/2
    right = (width + new_dimensions[1])/2
    bottom = (height + new_dimensions[0])/2

    return image.crop((left, top, right, bottom))

def image_resizer(image, new_dimensions):
    return image.resize(new_dimensions)

def to_greyscale(image):
    return image.convert("L")

def to_blackwhite(input_image):
    # h, l = input_image.size
    # image = input_image.copy()
    #
    # print(image)
    # for i in range(h):
    #     for j in range(l):
    #         if image[i][j]<128:
    #             image[i][j] = 0
    #         else:
    #             image[i][j] = 1
    return input_image.convert("1")

# convert image to a matrix of 0 and 1 and to black and white
def tomatrix_bew(image):
    imarray = np.array(image.getdata(),np.uint8).reshape(image.size[1], image.size[0])
    #print(imarray)
    #imarray = np.array(image)
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
    matrix = np.array(image.getdata(),np.uint8).reshape(image.size[1], image.size[0])
    print(matrix)
    print(matrix.shape)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    newarray = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                newarray[i][j] = 1
    print(newarray)
    return newarray

def collectimages(finaldim, dir):
    i = 0
    entries = 0

    #number of files checking
    for file in os.listdir(dir):
        if file.endswith(".tiff"):
            entries +=1

    dataset = np.zeros((entries, finaldim[0]*finaldim[1]))

    for file in os.listdir(dir):
        if file.endswith(".tiff"):
            # print(file)
            newdir = dir + "/" + file
            # print(newdir)

            #cut images

            # PARTE DA METTERE A POSTO
            # im[i] = Image.open(newdir)
            # im[i].show()
            # imarray[i] = np.array(im)
            # print(imarray[i].shape)
            # print(im[i].size)
            # print(imarray[i])

            im = Image.open(newdir)
            #im.show()

            orig_dim = im.size
            if orig_dim[0]>100 or orig_dim[1]>100:
                imp = image_cropper(im, [100, 100])
                imp = image_resizer(imp, finaldim)
            else:
                imp = image_resizer(im, [40, 70])
                print(imp.size)
                imp = image_resizer(im, finaldim)
                print(imp.size)
            imp = to_greyscale(imp)
            # print(imp)
            #imp.show()

            #imarray = tomatrix_bew(imp)

            imp = to_blackwhite(imp)
            #imp.show()
            imarray = tomatrix(imp)

            # testt = np.array(imp)
            # finalimage = imarray.flatten()
            # print(imarray)
            dataset[i] = imarray.flatten()
            print(imarray)
            print(dataset[i].reshape(finaldim[1], finaldim[0]))

            # print(imarray.shape)
            # print(imp.size)
            # print(imarray)
            # print(testt)

            i += 1

    return dataset

