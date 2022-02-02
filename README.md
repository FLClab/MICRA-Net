# MICRA-Net

[![DOI](https://zenodo.org/badge/307747246.svg)](https://zenodo.org/badge/latestdoi/307747246)

The current repository contains the source code for the publication _MICRA-Net: MICRoscopy Analysis Neural Network to solve detection, classification, and segmentation from a single simple auxiliary task_. A preprint version of the paper is available <a href="https://www.biorxiv.org/content/10.1101/2021.06.29.448970v1">here</a>.

The datasets and models are available to download from the <a href="https://s3.valeria.science/flclab-micranet/index.html">MICRA-Net website</a>.

The README is divided into the following sections
- [Overview](#overview)
- [Documentation](#documentation)
- [Experiment](#experiment)
    - [Downloads](#downloads)
    - [Training](#training)
    - [Inference](#inference)
- [System requirements](#system-requirements)
  - [Hardware requirements](#harware-requirements)
  - [Software requirements](#software-requirements)
    - [Python dependencies](#python-dependencies)
    - [Installation guide](#installation-guide)
- [Citation](#citation)

<a id="overview"></a>
# Overview

High throughput quantitative analysis of microscopy images presents a challenge due to the complexity of the image content and the difficulty to retrieve precisely annotated datasets. In this repository we introduce a weakly-supervised MICRoscopy Analysis neural network (MICRA-Net) that can be trained on a simple main classification task using image-level annotations to solve multiple more complex auxiliary tasks, such as segmentation, detection, and enumeration.

MICRA-Net relies on the latent information embedded within a trained model to achieve performances similar to state-of-the-art fully-supervised learning. This learnt information is extracted from the network using gradient class activation maps, which are combined to generate precise feature maps of the biological structures of interest.

<a id="documentation"></a>
# Documentation

The source code is available to the users within the `src` folder. The subfolder contains the code developed for each of the datasets used in the paper.

The provided scripts are all written in Python. Hence, the user should jump to jump to the [software requirements section](#software-requirements) to validate/install the right version of Python and it's dependencies.

Within the `src` folder, we provide the `main.py` file which allows the user to download the datasets from source and sets the proper symlinks for the following steps to the downloaded `MICRA-Net` folder.

Each dataset folder contains at least :
- `data/` : Some images sampled from the testing set.
- `baseline/` : A folder containing the baselines and a `predict.py` file to infer on the same provided example images.
- `predict.py` : A script which can be used to infer the network on a subset of testing images.
- `network.py` : The MICRA-Net architecture in PyTorch.

We provide an example of training MICRA-Net from an `hdf5` file in `src/Actin`. In the `src/Actin` folder, we also provide training examples for a U-Net and Mask R-CNN baselines. These training examples can serve as building blocks for training on a new dataset. See the [training section](#training) below for a detailed procedure.

To facilitate the inference on the testing images, we created a `predict.py` within each subfolders. Please refer to the [inference section](#inference) below for a detailed procedure.

<a id="experiment"></a>
# Experiment

In the following we provide the user with some steps to allow training and inference of images using the provided models.

<a id="downloads"></a>
## Downloads

We provide a script `main.py` which allows to automatically download the models and data from source. Following the installation of Python (see [Software requirements](#software-requirements)), the user may launch the script
```bash
cd src
python main.py
```

This script downloads the models and datasets in the `~/Downloads/MICRA-Net` folder of the computer. This folder contains a _models_ and a _datasets_ folder.

_A more experienced user may download specific models and set the path accordingly within the folders._

__NOTE.__ We intentionally removed the Ilastik models from the download to reduce the size of the downloaded folder. The Ilastik models can be downloaded from the following links [Actin](https://s3.valeria.science/flclab-micranet/MICRA-Net/models-ilastik/ActinModelZoo-ilastik.hdf5) and [Cell Tracking Challenge](https://s3.valeria.science/flclab-micranet/MICRA-Net/models-ilastik/CTCModelZoo-ilastik.hdf5).

The _models_ folder contains each zoo models, where each zoo is composed of one instance of a trained model. The zoo models are `hdf5` files with the following file architecture
```python
"file" : {
    "architecture1" : {
        "model_name" : {
            "weights_a" : h5py.Dataset,
            "weights_b" : h5py.Dataset
        }
    },
    "architecture2" : {
        "model_name" : {
            "weights_a" : h5py.Dataset,
            "weights_b" : h5py.Dataset
        }
    },
    ...
}
```

The _datasets_ folder contains the F-actin dataset which is already split in training, validation, and testing.

<a id="training"></a>
## Training

We provide a training example within the F-actin dataset folder using the downloaded datasets. To train MICRA-Net for one epoch use the following
```bash
cd src/Actin
python train.py --dry-run
```

The `--dry-run` flag is used to test the training of the model. By default, the model will be saved in a `hdf5` file in the output folder (`~/Downloads/MICRA-Net/Results/dryrun/checkpoints.hdf5`). Training on a standard CPU (i7-7700) should require approximately 15 minutes per epoch.

The same procedure may be applied to train the baseline models : U-Net and Mask R-CNN.
```bash
cd src/Actin/baseline/<UNet OR MaskRCNN>
python train.py --dry-run
```

### Training from in-house data

The provided training example in `src/Actin` contains all the necessary building blocks to train MICRA-Net for a different in-house dataset, provided that the user makes some minor modifications to the `HDF5Dataset` class.

The structure of the dataset should be the following
```python
"file" : {
    "group1" : {
        "data" : h5py.Dataset,
        "label" : h5py.Dataset
    },
    "group2" : {
        "data" : h5py.Dataset,
        "label" : h5py.Dataset
    },
    ...
}
```

<a id="inference"></a>
## Inference

We provide a `predict.py` script for all provided models. In all cases, the script can be launched with the `--cuda` flag to increase the speed of computation. Navigate to the desired folder and launch the script
```bash
cd src/Actin
python predict.py --cuda
```

This script will use the images provided within the `data/` folder and create a `segmentation/` folder containing all of the predictions.

In some cases, the `predict.py` script can be called with a different supervision level (using the `--supervision [LEVEL]` flag). The user should refer to the provided script for specific details.

The user may use the trained model obtained from [training](#training) by changing the model path. Specifically, the line
```python
model_path = os.path.join(".", "MICRA-Net", "models", "ActinModelZoo.hdf5")
# should be replaced by
model_path = os.path.join(".", "MICRA-Net", "Results", "dryrun", "checkpoints.hdf5")
```

For the Ilastik models, we provided within the baseline folders a `convert.py` script. This script may be used to extract the Ilastik model from the `hdf5` file. We refer the user to the [Ilastik website](https://www.ilastik.org/) to download the software and instructions on how to use it.

<a id="system-requirements"></a>
# System requirements

<a id="hardware-requirements"></a>
## Hardware requirements

For inference, MICRA-Net requires a standard computer to run the scripts with sufficient RAM to load an image and network in memory.

For training MICRA-Net and other baselines in the `Actin` folder, a minimum of 16G of available RAM is required to load the data in memory. It is strongly recommended to have a graphical processing unit (GPU). With the default parameters, the current memory necessary on the GPU is less than 12G. This can be reduced by lowering the `batch_size` parameters of the models.

<a id="software-requirements"></a>
## Software requirements

<a id="os-requirements"></a>
### OS requirements

The source code was tested on macOS (10.15), Ubuntu (18.04 and 20.04), and Windows 10.

<a id="python-dependencies"></a>
### Python dependencies

The source code `MICRA-Net` relies on Python scientific librairies. The source code was tested in a Python 3.7 environnement. We provide a `requirements.txt` file to facilitate the installation of the necessary dependencies.

<a id="installation-guide"></a>
### Installation guide

Assuming the users have a working installation of Python on their computer (we recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/)), the users should create a new Python 3.7 environnement to avoid impacting on other file dependencies. The complete installation should be less than 15 minutes.

```bash
conda create -n micranet python=3.7
conda activate micranet
pip install -r requirements.txt
```

<a id="citation"></a>
# Citation

If you use any of the material provided within the repository, please cite the following paper.

> _MICRA-Net: MICRoscopy Analysis Neural Network to solve detection, classification, and segmentation from a single simple auxiliary task_ Anthony Bilodeau, Constantin V.L. Delmas, Martin Parent, Paul De Koninck, Audrey Durand, Flavie Lavoie-Cardinal bioRxiv 2021.06.29.448970; doi: https://doi.org/10.1101/2021.06.29.448970

```
@article {Bilodeau2021,
	author = {Bilodeau, Anthony and Delmas, Constantin V.L. and Parent, Martin and De Koninck, Paul and Durand, Audrey and Lavoie-Cardinal, Flavie},
	title = {MICRA-Net: MICRoscopy Analysis Neural Network to solve detection, classification, and segmentation from a single simple auxiliary task},
	elocation-id = {2021.06.29.448970},
	year = {2021},
	doi = {10.1101/2021.06.29.448970},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/06/30/2021.06.29.448970},
	eprint = {https://www.biorxiv.org/content/early/2021/06/30/2021.06.29.448970.full.pdf},
	journal = {bioRxiv}
}

```

# License

License for Non-Commercial Use

If this software is redistributed, this license must be included.
The term software includes any source files, documentation, executables,
models, and data.

This software is available for general use by academic or non-profit,
or government-sponsored researchers. This license does not grant the
right to use this software or any derivation of it for commercial activities.
For commercial use, please contact Flavie Lavoie-Cardinal at Université Laval
at flavie.lavoie-cardinal@cervo.ulaval.ca

This software comes with no warranty or guarantee of any kind. By using this
software, the user accepts full liability.
