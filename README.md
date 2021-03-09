# MICRA-Net

The current repository contains the source code for the publication _MICRA-Net: MICRoscopy Analysis Neural Network to solve detection, classification, and segmentation from a single simple auxiliary task_.

The README is divided into the following sections
- [Overview](#overview)
- [Documentation](#documentation)
- [System requirements](#system-requirements)
  - [Hardware requirements](#harware-requirements)
  - [Software requirements](#software-requirements)
    - [Python dependencies](#python-dependencies)
    - [Installation guide](#installation-guide)
- [Citation](#citation)

# Overview

High throughput quantitative analysis of microscopy images presents a challenge due to the complexity of the image content and the difficulty to retrieve precisely annotated datasets. In this repository we introduce a weakly-supervised MICRoscopy Analysis neural network (MICRA-Net) that can be trained on a simple main classification task using image-level annotations to solve multiple more complex auxiliary tasks, such as segmentation, detection, and enumeration.

MICRA-Net relies on the latent information embedded within a trained model to achieve performances similar to state-of-the-art fully-supervised learning. This learnt information is extracted from the network using gradient class activation maps, which are combined to generate precise feature maps of the biological structures of interest.

# Documentation

The source code is available to the users within the `src` folder. The subfolder contains the code developped for each of the datasets used in the paper.

Each dataset folder contains at least :
- `pretrained/` : A folder containing the pretrained network parameters and the training parameters used.
- `data/` : Some images sampled from the testing set.
- `baseline/` : A folder containing the pretrained baseline and a `predict.py` file to infer on the same provided example images.
- `predict.py` : A script which can be used to infer the network on a subset of testing images.
- `network.py` : The MICRA-Net architecture in PyTorch.

We provide an example of training MICRA-Net in `src/MNIST` or `src/Actin`. Following the installation of Python with its dependencies, the users can test the training of the network using the following lines:
```bash
cd src/<{MNIST,Actin}>
python train.py
```

To facilitate the inference on the testing images, we created a `predict.py` script which can be launched using the following:
```bash
python predict.py
```
The infered images will be output in a `segmentation/` folder.

# System requirements

## Hardware requirements

MICRA-Net requires a standard computer to run the scripts with sufficient RAM to load an image and network in memory. It is recommended to have a graphical processing unit (GPU).

## Software requirements

### OS requirements

The source code was tested on macOS (10.15) and Ubuntu (18.04 and 20.04). It should also work on Windows with a few `filepath` modifications.

### Python dependencies

The source code `MICRA-Net` relies on Python scientific librairies. The source code was tested in a Python 3.7 environnement. We provide a `requirements.txt` file to facilitate the installation of the necessary dependencies.

### Installation guide

Assuming the users have a working installation of Python on their computer (we recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/)), the users  should create a new Python 3.7 environnement to avoid impacting on other file dependencies. The complete installation should be less than 15 minutes.

```bash
conda create -n micranet python=3.7
pip install -r requirements.txt
```

# Citation

If you use any of the material provided within the repository, please cite the following paper.

> Bilodeau, A. et al. (2020) MICRA-Net: MICRoscopy Analysis Neural Network to solve detection, classification, and segmentation from a single simple auxiliary task.

```
@article{bilodeau2020,
  title={MICRA-Net: MICRoscopy Analysis Neural Network to solve detection, classification, and segmentation from a single simple auxiliary task},
  author={Bilodeau, Anthony and V.L. Delmas, Constantin and Parent, Martin and De Koninck, Paul and Durand, Audrey and Lavoie-Cardinal, Flavie},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

# License

The project is covered under the __GPL-3.0 License__
