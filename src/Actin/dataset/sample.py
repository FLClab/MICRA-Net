
import numpy
import os, glob
import json

from skimage import draw, io

DENDRITE = 2

def _read_poly(path, image_shape):
    """
    Reads and creates the labels from the .poly.txt files

    :param path: The path to the .poly.txt file
    :param metadata: The metadata of the images

    NOTE, 0 is rings, 1 is uncertain, 2 is dendrite, 3 is filaments
    """
    label = numpy.zeros((2, *image_shape))
    with open(path, "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    for row in data:
        l = int(row[0:1])
        if l != 2:
            coordinates = eval(row[2:])
            r, c = [], []
            for coord in coordinates:
                r.append(int(coord[1]))
                c.append(int(coord[0]))
            rr, cc = draw.polygon(r, c, shape=image_shape)
            # if l in [0, 1]:
            if l in [0]:
                label[0][rr, cc] = 1
            elif l in [3]:
                label[1][rr, cc] = 1
    return label

def convert2int(img):
    """This function converts an uint image to an int image.

    :param img: A 2D numpy array of type uint

    :returns : A 2D numpy array converted to int
    """
    if img.min() == 0:
        return img
    if img.dtype == "uint8":
        return img - 2**8 / 2
    elif img.dtype == "uint16":
        return img - 2**16 / 2
    elif img.dtype == "uint32":
        return img - 2**32 / 2
    else:
        return img

def get_dendrite_foreground(img):
    """Gets the foreground of the dendrite channel using a gaussian blur of
    sigma = 20 and the otsu threshold.

    :param img: A 3D numpy

    :returns : A binary 2D numpy array of the foreground
    """
    blurred = filters.gaussian(img[DENDRITE], sigma=20)
    blurred /= blurred.max()
    val = filters.threshold_otsu(blurred)
    return (blurred > val).astype(int)

def sample_random_crops(dataset, num=10):
    """
    Samples
    """
    image_names, label_names, polytxt_names = dataset["image"], dataset["label"], dataset["polytxt"]

    for image_name, label_name, polytxt_name in zip(image_names, label_names, polytxt_names):
        image_name = image_name.replace("/media/anthony/DATA/Dataset/",
                                        "/home-local/Actin-Dataset/Dataset/")
        image = io.imread(image_name)
        image = numpy.transpose(image, axes=(2, 0, 1))

        print(image.mean())


if __name__ == "__main__":

    dataset = json.load(open("training_01-04-19.json", "r"))
    sample_random_crops(dataset)
