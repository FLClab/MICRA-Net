
import numpy
import torch
import os
import glob
import json

from tqdm import tqdm
from skimage import io
from matplotlib import pyplot
from torch.utils.data import DataLoader

import network
import loader

class Predicter:
    """
    Class to make predictions on a set of images
    """
    def __init__(self, data_path, model_path, save_folder, cuda=False, load_from_cache=False):
        """
        Inits the predicter class

        :param data_path: A file path to the dataset
        :param model_path: A file path to the model
        :param save_folder: A file path where to save the files
        :param cuda: Wheter to use cuda
        :param load_from_cache: Wheter to load from cache
        """
        # Assigns member variables
        self.data_path = data_path
        self.model_path = model_path
        self.save_folder = save_folder
        self.cuda = cuda

        dataset = loader.Datasetter(self.data_path)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

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
                X = X.cuda()

            pred = self.model.forward(X)
            pred = pred.cpu().data.numpy()

            self.save_images(i, X.cpu().data.numpy(), pred)

    def load_network(self):
        """
        Loads the model from model_path
        """
        net_params, trainer_params = self.load(self.model_path)
        self.model = network.UNet(**trainer_params)
        self.model.eval()
        self.model.load_state_dict(net_params)
        if self.cuda:
            self.model = self.model.cuda()

    def save_images(self, batch, images, pred):
        """
        Saves the images, masks, local maps, precise, coarse and predictions
        """
        io.imsave(os.path.join(self.save_folder, "{}_image.tif".format(batch)), images.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_pred.tif".format(batch)), pred.astype(numpy.float32), check_contrast=False)

    def load(self, folder):
        """
        Loads a previous network and optimizer state
        """
        net_params = torch.load(os.path.join(folder, "params.net"), map_location=None if self.cuda else "cpu")
        trainer_params = json.load(open(os.path.join(folder, "trainer_params.json"), "r"))
        return net_params, trainer_params

if __name__ == "__main__":

    data_path = os.path.join("..", "data")
    model_path = os.path.join(".", "pretrained")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=False)
    predicter.predict()
