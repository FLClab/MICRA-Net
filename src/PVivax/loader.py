import numpy
import glob
import os
import torch

from skimage import io
from torch.utils.data import Dataset

def scale_intensity(image):
    """
    Scales the intensity of an image to the desired mean and std

    :param image: A `numpy.ndarray`
    :param mean: A `numpy.ndarray` of the desired mean
    :param std: A `numpy.ndarray` of the desired std

    :returns : A `numpy.ndarray` of the scaled image
    """
    input_mean, input_std = image.mean(axis=(1, 2)), image.std(axis=(1, 2))
    input_mean, input_std = input_mean[:, numpy.newaxis, numpy.newaxis], input_std[:, numpy.newaxis, numpy.newaxis]
    # return image / input_mean * mean
    return (image - input_mean) / input_std

class Datasetter(Dataset):
    def __init__(self, data_path):

        self.data = list(sorted(glob.glob(os.path.join(data_path, "*.tif"))))

    def __getitem__(self, index):

        x = io.imread(self.data[index]).astype(numpy.float32)
        x = numpy.transpose(x, axes=(2, 0, 1))

        return x

    def __len__(self):
        return len(self.data)
