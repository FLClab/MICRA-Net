
import numpy
import torch
import pickle
import os
import sys
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import io

import loader
import UNet

class Predicter:
    """
    Predicter object contains all the necessary methods to load a model and save
    the predictions.
    """
    def __init__(self, data_path, model_path, save_folder, cuda=False):
        """
        Inits the Predicter object

        :param data_path: A file path to the dataset
        """
        # Assign member variables
        self.save_folder = save_folder
        self.cuda = cuda

        # Create the dataloader
        dataset = loader.DatasetterFromFile(data_path)
        self.loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

        # Loads the model
        self.load_network(model_path, cuda)

    def predict(self):
        """
        Classifies the data from the dataloader and saves the images, masks,
        class activated maps and the precise and coarse segmentation
        """
        for i, (X) in enumerate(tqdm(self.loader, desc="Prediction")):

            # Puts the data on the GPU
            X = X.unsqueeze(1)
            if self.cuda:
                X = X.cuda()
                y = y.cuda()

            prediction = self.model.forward(X)
            prediction = prediction.cpu().data.numpy()

            prediction = numpy.argmax(prediction, axis=1)

            self.save_images(i, X.cpu().numpy().squeeze(), prediction)

    def classify(self):
        """
        Classifies the data from the dataloader. Saves the predictions in a file
        """
        self.model.grad = False
        output = {}
        for i, (X) in enumerate(tqdm(self.loader, desc="Classification")):

            # Puts the data on the GPU
            X = X.unsqueeze(1)
            if self.cuda:
                X = X.cuda()

            # Computes the predictions
            predictions = self.model.forward(X)

            output[str(i)] = (torch.sigmoid(predictions) > 0.5).cpu().data.numpy().tolist()

        json.dump(output, open(os.path.join(self.save_folder, "predictions.json"), "w"), indent=4, sort_keys=True)
        self.model.grad = True

    def load_network(self, model_path, cuda):
        """
        Loads the network from the given folder in memory.

        :param model_path: The path to the model
        :param cuda: Wheter to put the model on the GPU
        """
        net_params, trainer_params = self.load(model_path, cuda)

        self.model = UNet.UNet(**trainer_params)
        self.model.eval()
        self.model.load_state_dict(net_params)
        if cuda:
            self.model = self.model.cuda()

    def save_images(self, batch, images, prediction):
        """
        Saves the images, local maps and the segmented images
        """
        io.imsave(os.path.join(self.save_folder, "{}_image.tif".format(batch)), images.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_prediction.tif".format(batch)), prediction.astype(numpy.float32), check_contrast=False)

    @staticmethod
    def load(model_path, cuda):
        """
        Loads a previous network model from the given folder. This folder should contain
        the files : params.net, optimizer.data and statsCkpt.pkl.

        :param model_path: The path of the folder containing the network

        :returns : The parameters of the network
        """
        net_params = torch.load(os.path.join(model_path, "params.net"), map_location=None if cuda else "cpu")
        trainer_params = json.load(open(os.path.join(model_path, "trainer_params.json"), "r"))
        return net_params, trainer_params

    @staticmethod
    def unflatten(arr):
        """
        Unflattens a flatten array with argmax

        :param arr: Numpy array

        :returns : The unflatten numpy array
        """
        arr = arr.astype(int)
        return numpy.transpose(numpy.eye(11)[arr], axes=(0, 3, 1, 2))


if __name__ == "__main__":

    data_path = os.path.join("..", "data")
    model_path = os.path.join(".", "pretrained")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=False)
    predicter.predict()
    predicter.classify()
