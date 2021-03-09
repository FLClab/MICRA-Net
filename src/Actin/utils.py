
import numpy
import random
import pickle
import os
import glob
import json

from collections import defaultdict
from skimage import draw, morphology, filters
from matplotlib import pyplot

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

def bin_image(ary, m, method=numpy.mean):
    """
    Bins every n x n pixels in a 2D numpy array

    :param ary: A 2D numpy array
    :param m: The shape of the binning (m x m)

    :returns : The binned numpy array
    """
    h, w = ary.shape
    return method(method(ary.reshape(h // m, m, w // m, m), axis=3), axis=1)

def get_foreground(ary):
    """
    Retreives the foreground of the input image. This thresholding computes the
    50% quantile of the image, removes small objects and fills small holes.

    :param ary: A 2D numpy array of the image

    :returns : A 2D binary numpy array of the image
    """
    binned = bin_image(ary, 8)
    ary = (binned > numpy.quantile(binned, q=0.5))
    return ary.repeat(8, axis=0).repeat(8, axis=1)
