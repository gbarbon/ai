__author__ = 'gbarbon'

from PIL import Image
import os
import numpy as np

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

def toarray(image):
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

            imp = image_cropper(im, [100, 100])
            imp = image_resizer(imp, [finaldim[0], finaldim[1]])
            imp = to_greyscale(imp)
            # print(imp)
            #imp = to_blackwhite(imp)
            # imp.show()

            imarray = toarray(imp)
            testt = np.array(imp)
            finalimage = imarray.flatten()
            dataset[i] = imarray.flatten()

            # print(imarray.shape)
            # print(imp.size)
            # print(imarray)
            # print(testt)

            i += 1

    return dataset

