
import numpy
import glob
import os
import torch

from skimage import io
from torch.utils.data import Dataset

class Datasetter(Dataset):
    def __init__(self, data_path, trainer_params):

        self.trainer_params = trainer_params
        self.data = list(sorted(glob.glob(os.path.join(data_path, "*.tif"))))

    def __getitem__(self, index):

        size = self.trainer_params["size"]
        x = io.imread(self.data[index]).astype(numpy.float32)
        samples = []
        for j in range(0, x.shape[0], size):
            for i in range(0, x.shape[1], size):
                samples.append(x[j : j + size, i : i + size])
        return numpy.array(samples)

    def __len__(self):
        return len(self.data)
