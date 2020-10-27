
import numpy
import torch
import os
import shutil
import time
import pickle
import json
import glob
import sys

from matplotlib import pyplot
from skimage import io, filters, morphology, measure
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

import loader
import gradCAM

from network import MICRANet

def threshold(raw_images):
    """
    Thresholds a stack of input images

    :param image: A `numpy.ndarray` of size (B, C, H, W)

    :returns : A `numpy.ndarray` of the thresholded images
    """
    raw_images = numpy.pad(raw_images, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")

    predictions = []
    for raw_image in raw_images:

        raw_image = raw_image.squeeze()
        maximum_pred_pos = numpy.unravel_index(raw_image.argmax(), raw_image.shape)

        filtered = filters.gaussian(raw_image, sigma=10)

        prediction = numpy.zeros_like(filtered)

        if not numpy.all(filtered == 0):

            threshold = filters.threshold_otsu(filtered)
            filtered = filtered >= threshold

            filtered = morphology.remove_small_holes(filtered, area_threshold=2500)
            precise_label_boundary, num_labels = measure.label(filtered, return_num=True)
            precise_label = measure.label(filtered)

            if num_labels < 1:
                continue
            for j in range(1, num_labels + 1):
                sorted_vertices = measure.find_contours((precise_label == j).astype(float), 0.5, fully_connected="high")
                if sorted_vertices:
                    for vertices in sorted_vertices:
                        in_poly = measure.points_in_poly(numpy.array(maximum_pred_pos)[numpy.newaxis, :], vertices)
                        if in_poly:
                            # ax[2].imshow((precise_label == j).astype(int)[1 : -1, 1 : -1])
                            prediction = (precise_label == j).astype(int)[1 : -1, 1 : -1]
                            break
            predictions.append(prediction)
    return numpy.array(predictions)

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
        self.loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

        # Loads the model
        self.load_network()

    def predict(self):
        """
        Predicts the data from the dataloader. Saves the output save_folder
        """
        for i, (X) in enumerate(tqdm(self.loader, desc="Loader")):
            if X.ndim == 3:
                X = X.unsqueeze(1)
            if self.cuda:
                X  = X.cuda()

            raw_avg = numpy.zeros((len(X), self.trainer_params["num_classes"], X.shape[-2], X.shape[-1]))
            for j, model in enumerate(tqdm(self.models, desc="Models", leave=False)):
                if self.cuda:
                    model = model.cuda()

                # Computes the gradient class activation map
                local_maps, pred = gradCAM.class_activation_map(model, X, cuda=self.cuda, size=X.shape[-1])
                pred = pred >= self.class_thresholds

                # Thresholds the class activation map
                _, raw_precise = gradCAM.segment_cam(local_maps, pred, X, size=X.shape[-1])

                raw_avg += raw_precise

            raw_precise = raw_avg / len(self.models)
            precise = threshold(raw_precise)

            self.save_images(f"{i}", X.cpu().numpy().squeeze(),
                                local_maps, precise, raw_precise)

    def classify(self):
        """
        Classifies the data from the dataloader. Saves the output to a json file
        """
        output = {}
        for i, (X) in enumerate(tqdm(self.loader, desc="Loader")):
            if X.ndim == 3:
                X = X.unsqueeze(1)
            if self.cuda:
                X  = X.cuda()

            predictions = []
            for j, model in enumerate(tqdm(self.models, desc="Models", leave=False)):
                model.grad = False

                if self.cuda:
                    model = model.cuda()

                preds = model.forward(X)
                predictions.append(torch.sigmoid(preds).cpu().data.numpy())

                model.grad = True

            predictions = numpy.array(predictions) >= self.class_thresholds
            output[str(i)] = predictions.tolist()

        json.dump(output, open(os.path.join(self.save_folder, "predictions.json"), "w"), indent=4, sort_keys=True)

    def load_network(self):
        """
        Loads the model from model_path
        """
        net_params = self.load(self.model_path)

        self.models = [MICRANet(grad=True, **self.trainer_params) for _ in range(len(net_params))]
        for model in self.models:
            model.eval()
        for model, param in zip(self.models, net_params):
            model.load_state_dict(param)

    def save_images(self, batch, images, local_maps, precise, raw_precise):
        """
        Saves the images, masks, local maps, precise, coarse and predictions
        """
        io.imsave(os.path.join(self.save_folder, "{}_image.tif".format(batch)), images.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_localmap.tif".format(batch)), local_maps.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_precise.tif".format(batch)), precise.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_rawprecise.tif".format(batch)), raw_precise.astype(numpy.float32), check_contrast=False)

    def load(self, folder):
        """
        Loads a previous network and optimizer state
        """
        self.trainer_params = json.load(open(os.path.join(folder, "trainer_params.json"), "r"))
        self.trainer_params["size"] = 512
        self.class_thresholds = numpy.load(os.path.join(folder, "class_thresholds.npy"))
        if ("k-folds" in self.trainer_params) and (self.trainer_params["k-folds"] != 1):
            net_params = [torch.load(os.path.join(folder, f"fold-{i}", "params.net"), map_location=None if self.cuda else "cpu") for i in range(self.trainer_params["k-folds"])]
        return net_params

if __name__ == "__main__":

    data_path = os.path.join(".", "data")
    model_path = os.path.join(".", "pretrained")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=False)
    predicter.predict()
    predicter.classify()
