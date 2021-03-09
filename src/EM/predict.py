
import numpy
import torch
import os
import shutil
import time
import pickle
import json
import glob
import sys
import h5py

from matplotlib import pyplot
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

import loader
import gradCAM

from network import MICRANet

class Predicter:
    """
    Class to make predictions on a set of images
    """
    def __init__(self, data_path, model_path, save_folder, cuda=False, size=512,
                 supervision="1:5"):
        """
        Inits the predicter class

        :param data_path: A file path to the dataset
        :param model_path: A file path to the model
        :param save_folder: A file path where to save the files
        :param cuda: Wheter to use cuda
        :param supervision: A `str` of the level of supervision
        """
        # Assigns member variables
        self.data_path = data_path
        self.model_path = model_path
        self.save_folder = save_folder
        self.cuda = cuda
        self.size = size
        self.supervision = supervision

        dataset = loader.Datasetter(self.data_path)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

        # Loads the model
        self.load_network()

    def predict(self):
        """
        Predicts the data from the dataloader. Saves the output save_folder
        """
        for i, (X) in enumerate(self.loader):

            if X.ndim == 3:
                X = X.unsqueeze(1)

            if self.cuda:
                X  = X.cuda()

            # Computes the gradient class activation map
            local_maps, pred = gradCAM.class_activation_map(self.model, X, cuda=self.cuda, size=X.shape[-1])
            pred = pred >= self.class_thresholds

            # Thresholds the class activation map
            precise, raw_precise = gradCAM.segment_cam(local_maps, pred, X, size=X.shape[-1])

            self.save_images(i, X.cpu().numpy().squeeze(),
                                local_maps, precise, raw_precise)

    def classify(self):
        """
        Classifies the data from the dataloader. Saves the output to a json file
        """
        self.model.grad = False # We do not need the gradients in classification
        output = {}
        for i, (X) in tqdm(enumerate(self.loader)):

            if X.ndim == 3:
                X = X.unsqueeze(1)
            if self.cuda:
                X = X.cuda()

            # Make the predictions
            predictions = self.model.forward(X)

            output[str(i)] = [(torch.sigmoid(predictions).cpu().data.numpy() >= self.class_thresholds).tolist()]

        json.dump(output, open(os.path.join(self.save_folder, "predictions.json"), "w"), indent=4, sort_keys=True)
        self.model.grad = True

    def load_network(self):
        """
        Loads the model from model_path
        """
        net_params, trainer_params = self.load()
        self.model = MICRANet(grad=True, **trainer_params)
        self.model.eval()
        self.model.load_state_dict(net_params)
        if self.cuda:
            self.model = self.model.cuda()

    def save_images(self, batch, images, local_maps, precise, raw_precise):
        """
        Saves the images, masks, local maps, precise, coarse and predictions
        """
        io.imsave(os.path.join(self.save_folder, "{}_image.tif".format(batch)), images.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_localmap.tif".format(batch)), local_maps.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_precise.tif".format(batch)), precise.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_rawprecise.tif".format(batch)), raw_precise.astype(numpy.float32), check_contrast=False)

    def load(self):
        """
        Loads a previous network and optimizer state
        """
        with h5py.File(self.model_path, "r") as file:
            networks = {}
            for key, values in file["-".join(("MICRANet", self.supervision))].items():
                networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
            trainer_params = json.loads(file["-".join(("MICRANet", self.supervision))].attrs["trainer_params"])
            self.class_thresholds = numpy.array(file["-".join(("MICRANet", self.supervision))].attrs["class_thresholds"])
        net_params = networks[key]
        return net_params, trainer_params

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="(optional) Wheter cuda can be used")
    parser.add_argument("--supervision", type=str, default="1:5",
                        help="(optional) Which supervision level to load")
    args = parser.parse_args()

    available_supervision = ["2:1", "1:1", "1:2", "1:5", "1:8", "1:16"]
    assert args.supervision in available_supervision, "The supervision level does not exists... Here's the valid list : [{}]".format(", ".join(available_supervision))

    data_path = os.path.join(".", "data")
    model_path = os.path.join(".", "MICRA-Net", "models", "EMModelZoo.hdf5")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=args.cuda, supervision=args.supervision)
    predicter.predict()
    predicter.classify()
