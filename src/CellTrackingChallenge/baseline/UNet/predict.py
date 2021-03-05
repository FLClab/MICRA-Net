
import numpy
import torch
import os
import json
import glob
import h5py

from matplotlib import pyplot
from skimage import io
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader

from UNet import UNet

class Predicter:
    """
    Class to make predictions on a set of images
    """
    def __init__(self, data_path, model_path, save_folder, cuda=False, size=512,
                 supervision="FS"):
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
        self.supervision = supervision

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
        net_params, trainer_params = self.load()
        self.model = UNet(**trainer_params)
        if trainer_params["model_params"]["num_classes"] != 2:
            self.model.conv_u0dscore = nn.Conv2d(in_channels=128, out_channels=trainer_params["model_params"]["num_classes"], kernel_size=3, padding=1)
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

    def load(self):
        """
        Loads a previous network and optimizer state
        """
        with h5py.File(os.path.join(self.model_path, "CTCModelZoo.hdf5"), "r") as file:
            networks = {}
            for key, values in file["-".join(("UNet", self.supervision))].items():
                networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
            trainer_params = json.loads(file["-".join(("UNet", self.supervision))].attrs["trainer_params"])
        net_params = networks[key]
        return net_params, trainer_params

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="(optional) Wheter cuda can be used")
    parser.add_argument("--supervision", type=str, default="FS",
                        help="(optional) Which supervision level to load")
    args = parser.parse_args()

    available_supervision = ["FS", "ALT2", "ALT5", "ALT10", "BBOX"]
    assert args.supervision in available_supervision, "The supervision level does not exists... Here's the valid list : [{}]".format(", ".join(available_supervision))

    data_path = os.path.join("..", "..", "data")
    model_path = os.path.join("..", "..", "pretrained")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=args.cuda, supervision=args.supervision)
    predicter.predict()
