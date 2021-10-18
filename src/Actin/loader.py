
import numpy
import glob
import os
import torch
import random
import h5py

from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

class HDF5Dataset(Dataset):
    """
    Creates a `Dataset` from a HDF5 file. It loads all the HDF5 file in cache. This
    increases the loading speed.

    :param file_path: A `str` to the hdf5 file
    :param data_aug: A `float` in range [0, 1]
    :param validation: (optional) Wheter the Dataset is for validation (no data augmentation)
    :param size: (optional) The size of the crops
    :param step: (optional) The step between each crops
    """
    def __init__(self, file_path, data_aug=0, validation=False, size=256, step=0.75, **kwargs):
        super(HDF5Dataset, self).__init__()

        self.file_path = file_path
        self.size = size
        self.step = step
        self.validation = validation
        self.data_aug = data_aug

        self.cache = {}

        self.samples = self.generate_valid_samples()

    def generate_valid_samples(self):
        """
        Generates a list of valid samples from the dataset. This is performed only
        once at each training
        """
        samples = []
        with h5py.File(self.file_path, "r") as file:
            for group_name, group in tqdm(file.items(), desc="Groups", leave=False):
                data = group["data"][()].astype(numpy.float32) # Images
                label = group["label"][()] # shape is Rings, Fibers, and Dendrite
                shapes = group["label"].attrs["shapes"] # Not all images have same shape
                for k, (dendrite_mask, shape) in enumerate(zip(label[:, -1], shapes)):
                    for j in range(0, shape[0], int(self.size * self.step)):
                        for i in range(0, shape[1], int(self.size * self.step)):
                            dendrite = dendrite_mask[j : j + self.size, i : i + self.size]
                            if dendrite.sum() >= 0.1 * self.size * self.size: # dendrite is at least 1% of image
                                samples.append((group_name, k, j, i))
                self.cache[group_name] = {"data" : data, "label" : label[:, :-1]}
        return samples

    def __getitem__(self, index):
        """
        Implements the `__getitem__` function of the `Dataset`

        :param index: An `int` of the sample to return

        :returns: A `torch.tensor` of the image
                  A `torch.tensor` of the label
        """
        group_name, k, j, i = self.samples[index]

        image_crop = self.cache[group_name]["data"][k, j : j + self.size, i : i + self.size]
        label_crop = self.cache[group_name]["label"][k, :, j : j + self.size, i : i + self.size]

        if image_crop.size != self.size*self.size:
            image_crop = numpy.pad(image_crop, ((0, self.size - image_crop.shape[0]), (0, self.size - image_crop.shape[1])), "constant")
            label_crop = numpy.pad(label_crop, ((0, 0), (0, self.size - label_crop.shape[1]), (0, self.size - label_crop.shape[2])), "constant")

        image = image_crop.astype(numpy.float32)
        label = numpy.sum(label_crop > 0, axis=(1, 2)) > (0.05 * self.size * self.size)

        # Applies data augmentation
        if not self.validation:
            if random.random() < self.data_aug:
                # left-right flip
                image = numpy.fliplr(image).copy()

            if random.random() < self.data_aug:
                # up-down flip
                image = numpy.flipud(image).copy()

            if random.random() < self.data_aug:
                # intensity scale
                intensityScale = numpy.clip(numpy.random.lognormal(0.01, numpy.sqrt(0.01)), 0, 1)
                image = numpy.clip(image * intensityScale, 0, 1)

            if random.random() < self.data_aug:
                # gamma adaptation
                gamma = numpy.clip(numpy.random.lognormal(0.005, numpy.sqrt(0.005)), 0, 1)
                image = numpy.clip(image**gamma, 0, 1)

        x = torch.tensor(image_crop, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.samples)

class Datasetter(Dataset):
    def __init__(self, data_path):

        self.data = glob.glob(os.path.join(data_path, "*.tif"))

    def __getitem__(self, index):

        x = io.imread(self.data[index]).astype(numpy.float32)

        return x

    def __len__(self):
        return len(self.data)
