
import numpy
import random
import pickle
import os
import glob
import json

from collections import defaultdict
from skimage import draw, morphology, filters
from matplotlib import pyplot


def retreive_data(path, with_masks=False):
    """
    Retreives the data from the folder. The images are tifffile while the labels
    are contained in an excel file.

    :param path: The path of the folder

    :returns : A list of X and a list of y filenames
    """
    def order(X, y):
        X_short = [item.split(os.path.sep)[-1].split(".msr")[0] for item in X]
        y_short = [item.split("spots_")[-1].split(".msr")[0] for item in y]
        return [y[y_short.index(item)] for item in X_short]
    X = glob.glob(os.path.join(path, "*_merged.tif" if not with_masks else "*_masked.tif"))
    y = order(X, glob.glob(os.path.join(path, "*.xlsx")))
    return X, y

def save_json(path, d):
    """
    Saves the json file to the output path

    :param path: The complete path where to save the json file (including filename)
    :param d: The dict to save to the file
    """
    json.dump(d, open(path, "w"), sort_keys=True, indent=4)

def load_json(path):
    """
    Loads the json file from the path

    :param path: The complete path of the json file

    :returns : json dict
    """
    return json.load(open(path, "r"))

def split_train_validation(data, ratio=0.7):
    """
    Splits the data in train and validation with the good ratio

    :param data: A dict of the data
    :param ratio: The ratio of the split

    :returns : A dict for training and for validation
    """
    training, validation = defaultdict(list), defaultdict(list)
    for key, values in data.items():
        random.shuffle(values)
        split = int(len(values) * ratio)
        training[key] = values[:split]
        validation[key] = values[split:]
    return training, validation

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


def bin_image(ary, m, method=numpy.mean):
    """
    Bins every n x n pixels in a 2D numpy array

    :param ary: A 2D numpy array
    :param m: The shape of the binning (m x m)

    :returns : The binned numpy array
    """
    h, w = ary.shape
    return method(method(ary.reshape(h // m, m, w // m, m), axis=3), axis=1)

def binary_ops(ary):
    """
    Applies a list of binary operations to the input array

    :param ary: A 2D numpy boolean array (converted internally)

    :returns : A 2D numpy array with applied binary operations
    """
    from matplotlib import pyplot
    fig, axes = pyplot.subplots(2, 1)
    axes[0].imshow(ary)
    ary = morphology.binary_closing(ary)
    ary = morphology.binary_opening(ary, selem=morphology.square(3))
    axes[1].imshow(ary)
    pyplot.show()
    return ary.astype(bool)


def get_foreground(ary):
    """
    Retreives the foreground of the input image. This thresholding computes the
    50% quantile of the image, removes small objects and fills small holes.

    :param ary: A 2D numpy array of the image

    :returns : A 2D binary numpy array of the image
    """
    binned = bin_image(ary, 8)
    # qs = numpy.arange(0., 1, 0.1)
    # sigmas = [0, 0.1, 1, 2, 3]
    # fig, axes = pyplot.subplots(len(sigmas), len(qs), figsize=(15,8), sharex=True, sharey=True)
    # for j, sig in enumerate(sigmas):
    #     filtered = filters.gaussian(binned, sigma=sig)
    #     filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
    #     for i, q in enumerate(qs):
    #         arr = filtered > numpy.quantile(filtered, q=q)
    #         arr = morphology.remove_small_objects(arr, min_size=128)
    #         arr = morphology.remove_small_holes(arr, area_threshold=128)
    #         axes[j, i].imshow(arr)

    # ary = filters.gaussian(ary, sigma=0.5)
    # ary = (ary - ary.min()) / (ary.max() - ary.min())

    ary = (binned > numpy.quantile(binned, q=0.5))
    # ary = morphology.remove_small_objects(ary, min_size=128)
    # ary = morphology.remove_small_holes(ary, area_threshold=128)
    return ary.repeat(8, axis=0).repeat(8, axis=1)
