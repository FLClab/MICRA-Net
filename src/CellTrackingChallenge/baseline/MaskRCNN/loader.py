
import numpy
import glob
import os
import torch

from skimage import io
from torch.utils.data import Dataset

class PredictionBuilder:
    """
    This class is used to create the final prediction from the predictions
    that are infered by the network. This class stores the predictions in an output
    array to avoid memory overflow with the method `add_predictions` and then
    computes the mean prediction of the overlap with the `return_prediction` method.

    :param shape: The shape of the image
    :param size: The size of the crops
    :param num_classes: The number of classes
    """
    def __init__(self, shape, size, num_classes=2):
        # Assign member variables
        self.shape = shape
        self.size = size

        # Creates the output arrays
        self.pred = numpy.zeros((num_classes, self.shape[0] + self.size, self.shape[1] + self.size), dtype=numpy.float32)
        self.pixels = numpy.zeros((self.shape[0] + self.size, self.shape[1] + self.size), dtype=numpy.float32)

    def add_predictions_ji(self, prediction, j, i):
        """
        Method to store the predictions in the output array at the corresponding
        position. We suppose a central postion of the crop

        :param predictions: A `numpy.ndarray` of prediction with size (features, H, W)
        :param j: An `int` of the row position
        :param i: An `int` of the column position
        """
        # Verifies the shape of prediction
        if prediction.ndim != 3:
            prediction = prediction[numpy.newaxis, ...]

        # Crops top and left part of image if necessary
        pred = prediction[:, 0 if j - self.size // 2 >= 0 else -1 * (j - self.size // 2) : ,
                              0 if i - self.size // 2 >= 0 else -1 * (i - self.size // 2) : ]

        # Stores prediction in output arrays
        self.pred[:, max(0, j - self.size // 2) : j + self.size // 2,
                     max(0, i - self.size // 2) : i + self.size // 2] += pred
        self.pixels[max(0, j - self.size // 2) : j + self.size // 2,
                    max(0, i - self.size // 2) : i + self.size // 2] += 1

    def return_prediction(self):
        """
        Method to return the final prediction.
        :returns : The average prediction map from the overlapping predictions
        """
        self.pixels[self.pixels == 0] += 1 # Avoids division by 0
        return (self.pred / self.pixels)[:, :self.shape[0], :self.shape[1]]

class ImageFolderIterator:
    """
    Creates an `ImageFolderIterator` object.

    This allows the user to itterate through all the images with the folder
    """
    def __init__(self, data_path):
        """
        Instantiates the `ImageFolderIterator`

        :param data_path: A `str` of the folder path
        """
        self.data = list(sorted(glob.glob(os.path.join(data_path, "*.tif"))))

    def __getitem__(self, index):
        return io.imread(self.data[index]).astype(numpy.float32)

    def __len__(self):
        return len(self.data)

class Datasetter(Dataset):
    def __init__(self, image, trainer_params):

        self.image = image
        self.trainer_params = trainer_params

        self.samples = self.generate_samples()

    def generate_samples(self):
        """
        Generates all samples in an image based on the shape of the image
        """
        samples = []
        size = self.trainer_params["size"]
        for j in range(0, self.image.shape[-2], size // 2):
            for i in range(0, self.image.shape[-1], size // 2):
                samples.append((j + size // 2, i + size // 2))
        return samples

    def __getitem__(self, index):

        size = self.trainer_params["size"]

        j, i = self.samples[index]
        _, m , n = self.image.shape
        x = self.image[:, max(0, j - size // 2) : j + size // 2,
                          max(0, i - size // 2) : i + size // 2]
        if numpy.prod(x.shape[-2:]) != size * size:
            x = numpy.pad(x, ((0, 0),
                              (0 if j - size // 2 >= 0 else -1 * (j - size // 2), (j + size // 2) - m if j + size // 2 > m else 0),
                              (0 if i - size // 2 >= 0 else -1 * (i - size // 2), (i + size // 2) - n if i + size // 2 > n else 0)), mode="symmetric")
        x = torch.tensor(x)
        return x, (j, i)

    def __len__(self):
        return len(self.samples)
