
import numpy
import os
import glob

from skimage import io
from torch.utils.data import Dataset

class Datasetter(Dataset):
    def __init__(self, X, y, masks=None):
        self.data = X
        self.target = y
        self.masks = masks

    def __getitem__(self, index):
        x = self.data[index].astype(numpy.float32)
        y = self.target[index].astype(numpy.float32)
        if self.masks is not None:
            masks = self.masks[index].astype(numpy.float32)

        if self.masks is not None:
            return x, y, masks
        return x, y

    def __len__(self):
        return len(self.data)


class DatasetterFromFile(Dataset):
    def __init__(self, data_path):

        self.data = list(sorted(glob.glob(os.path.join(data_path, "*.tif"))))

    def __getitem__(self, index):

        x = io.imread(self.data[index]).astype(numpy.float32)

        return x

    def __len__(self):
        return len(self.data)
