
import numpy
import torch
import os
import json
import glob

from matplotlib import pyplot
from skimage import io
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader
import merge

from UNet import UNet

class Predicter:
    """
    Class to make predictions on a set of images
    """
    def __init__(self, data_path, model_path, save_folder, cuda=False, size=512):
        """
        Inits the predicter class

        :param data_path: A file path to the dataset
        :param model_path: A file path to the model
        :param save_folder: A file path where to save the files
        :param cuda: Wheter to use cuda
        """
        # Assigns member variables
        self.data_path = data_path
        self.model_path = model_path
        self.save_folder = save_folder
        self.cuda = cuda
        self.size = size

        dataset = loader.Datasetter(self.data_path)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

        # Loads the model
        self.load_network()

    def predict(self):
        """
        Predicts the data from the dataloader. Saves the output save_folder
        """
        for i, (X) in enumerate(tqdm(self.loader)):

            if X.ndim == 3:
                X = X.unsqueeze(1)
            if self.cuda:
                X  = X.cuda()

            pred = self.model.forward(X)
            pred = pred.cpu().data.numpy()
            pred = numpy.argmax(pred, axis=1)

            self.save_images(i, X.cpu().numpy().squeeze(),
                                pred)

    def load_network(self):
        """
        Loads the model from model_path
        """
        net_params, trainer_params = self.load(self.model_path)
        self.model = UNet(**trainer_params)
        if trainer_params["num_classes"] != 2:
            self.model.conv_u0dscore = nn.Conv2d(in_channels=128, out_channels=trainer_params["num_classes"], kernel_size=3, padding=1)
        self.model.eval()
        self.model.load_state_dict(net_params)
        if self.cuda:
            self.model = self.model.cuda()

    def save_images(self, batch, images, precise):
        """
        Saves the images, masks, local maps, precise, coarse and predictions
        """
        io.imsave(os.path.join(self.save_folder, "{}_image.tif".format(batch)), images.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_precise.tif".format(batch)), precise.astype(numpy.float32), check_contrast=False)

    def load(self, folder):
        """
        Loads a previous network and optimizer state
        """
        net_params = merge.load(os.path.join(folder, "params.net"), map_location=None if self.cuda else "cpu")
        trainer_params = json.load(open(os.path.join(folder, "trainer_params.json"), "r"))
        return net_params, trainer_params

if __name__ == "__main__":

    data_path = os.path.join("..", "data")
    model_path = os.path.join(".", "pretrained")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=False)
    predicter.predict()
