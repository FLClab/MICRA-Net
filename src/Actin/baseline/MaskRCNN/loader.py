
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
                for k, (_label, shape) in enumerate(zip(label, shapes)):
                    for j in range(0, shape[0], int(self.size * self.step)):
                        for i in range(0, shape[1], int(self.size * self.step)):
                            positive_label = _label[:2, j : j + self.size, i : i + self.size] > 0
                            if positive_label.sum() >= 0.01 * self.size * self.size: # MaskRCNN requires positive crops
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

        image_crop = image_crop.astype(numpy.float32)
        label_crop = label_crop.astype(numpy.float32)

        # Applies data augmentation
        if not self.validation:
            if random.random() < self.data_aug:
                # left-right flip
                image_crop = numpy.fliplr(image_crop).copy()
                label_crop = numpy.fliplr(label_crop).copy()

            if random.random() < self.data_aug:
                # up-down flip
                image_crop = numpy.flipud(image_crop).copy()
                label_crop = numpy.flipud(label_crop).copy()

            if random.random() < self.data_aug:
                # intensity scale
                intensityScale = numpy.clip(numpy.random.lognormal(0.01, numpy.sqrt(0.01)), 0, 1)
                image_crop = numpy.clip(image_crop * intensityScale, 0, 1)

            if random.random() < self.data_aug:
                # gamma adaptation
                gamma = numpy.clip(numpy.random.lognormal(0.005, numpy.sqrt(0.005)), 0, 1)
                image_crop = numpy.clip(image_crop**gamma, 0, 1)

        boxes, labels, masks = [], [], []
        for class_id, mask in enumerate(label_crop):
            # Returns unique ids from mask
            obj_ids = numpy.unique(mask)[1:]
            # Slices each object in the mask in a new array
            obj_mask = mask == obj_ids[:, numpy.newaxis, numpy.newaxis]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            if num_objs > 0:
                keep = []
                for i in range(num_objs):
                    pos = numpy.where(obj_mask[i])
                    xmin = numpy.min(pos[1])
                    xmax = numpy.max(pos[1])
                    ymin = numpy.min(pos[0])
                    ymax = numpy.max(pos[0])
                    # Verify if valid box
                    if ((xmax - xmin) < 4) or ((ymax - ymin) < 4):
                        continue
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id + 1)
                    keep.append(i)
                masks.append(obj_mask[keep])

        masks = numpy.concatenate(masks, axis=0)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = []

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        x = torch.tensor(image_crop, dtype=torch.float32)
        return x, target

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
