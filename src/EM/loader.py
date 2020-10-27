
import numpy
import glob
import os
import torch

from skimage import io
from torch.utils.data import Dataset

class Datasetter(Dataset):
    def __init__(self, data_path):

        self.data = list(sorted(glob.glob(os.path.join(data_path, "*.tif"))))

    def __getitem__(self, index):

        x = io.imread(self.data[index]).astype(numpy.float32)
        x = x / 255.
        
        return x

    def __len__(self):
        return len(self.data)
