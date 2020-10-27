
import torch
import os
import glob

from collections import OrderedDict

def save(params, savepath):
    """
    Saves the parameters of a network by splitting the dict params

    :param params: An `OrderedDict` of parameters
    :param savepath: A `str` where to save the parameters
    """
    os.makedirs(savepath, exist_ok=True)
    for key, params in params.items():
        path = os.path.join(savepath, key)
        torch.save(params, path)

def load(params_path, map_location="cpu"):
    """
    Loads the parameters of a network by combining files from a folder

    :param params_path: A `str` where to look for the file

    :returns : An `OrderedDict` of the network parameters
    """
    if os.path.isfile(params_path):
        return torch.load(params_path, map_location=map_location)
    # Assumes it is a folder
    odict = OrderedDict()
    for file in glob.glob(os.path.join(params_path, "*")):
        params = torch.load(file, map_location=map_location)
        key = os.path.basename(file)
        odict[key] = params

    return odict

if __name__ == "__main__":

    # params = torch.load("pretrained/params_.net")
    # save(params, "pretrained/params.net")
    load("pretrained/params.net")
