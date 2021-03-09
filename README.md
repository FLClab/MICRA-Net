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

The source code is available to the users within the `src` folder. The subfolder contains the code developed for each of the datasets used in the paper.

Within the `src` folder, we provide the `main.py` file which allows the user to download the datasets from source and sets the proper symlinks for the following steps to the downloaded `MICRA-Net` folder.

Each dataset folder contains at least :
- `data/` : Some images sampled from the testing set.
- `baseline/` : A folder containing the baselines and a `predict.py` file to infer on the same provided example images.
- `predict.py` : A script which can be used to infer the network on a subset of testing images.
- `network.py` : The MICRA-Net architecture in PyTorch.

We provide an example of training MICRA-Net from a predefined `numpy.ndarray` in `src/MNIST` or from an `hdf5` file in `src/Actin`. We recommend using the latter when training MICRA-Net. In the `src/Actin` folder, we also provide training examples for a U-Net and Mask R-CNN baselines. These training examples can serve to train on a new dataset. See the [training section](#training) below for a detailed procedure.

To facilitate the inference on the testing images, we created a `predict.py` within each subfolders. Please refer to the [inference section](#inference) below for a detailed procedure.

# Experiment

In the following we provide the user with some steps to allow training and inference of images using the provided models.

## Downloads

We provide a script `main.py` which allows to download the models and data from source. Following the installation of Python (see [Software requirements](#software-requirements)), the user may launch the script
```bash
cd src
python main.py
```

This script downloads the models and datasets in the `~/Downloads/MICRA-Net` folder of the computer. This folder contains a _models_ and a _datasets_ folder. __NOTE__ We intentionally removed the Ilastik models from the download to reduce the size of the download folder. The Ilastik models can be downloaded from here ([Actin](https://s3.valeria.science/flclab-micranet/MICRA-Net/models-ilastik/ActinModelZoo-ilastik.hdf5) and [Cell Tracking Challenge](https://s3.valeria.science/flclab-micranet/MICRA-Net/models-ilastik/CTCModelZoo-ilastik.hdf5)).

The _models_ folder contains each zoo models, where each zoo is composed of one instance of a trained model. The zoo models are `hdf5` files with the following file architecture
```
FILE : {
    "ARCHITECTURE" : {
        "model_name" : {
            "weights_a" : [],
            "weights_b" : []
        }
    },
    "ARCHITECTURE" : {
        "model_name" : {
            "weights_a" : [],
            "weights_b" : []
        }
    }
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

The `--dry-run` flag is used to test the training of the model. By default, the model will be saved in a `hdf5` file in the output folder (`~/Downloads/MICRA-Net/Results/dryrun/checkpoints.hdf5`).

The same procedure may be applied to train the baseline models : U-Net and Mask R-CNN.
```bash
cd src/Actin/baseline/<UNet OR MaskRCNN>
python train.py --dry-run
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

# System requirements

## Hardware requirements

For inference, MICRA-Net requires a standard computer to run the scripts with sufficient RAM to load an image and network in memory.

For training MICRA-Net and other baselines in the it Actin folder, a minimum of 16G of available RAM is required to load the data in memory. It is strongly recommended to have a graphical processing unit (GPU). With the default parameters, the current memory necessary on the GPU is less than 12G.

<a id="software-requirements"></a>
## Software requirements

### OS requirements

The source code was tested on macOS (10.15) and Ubuntu (18.04 and 20.04). It should also work on Windows with a few `filepath` modifications.

### Python dependencies

The source code `MICRA-Net` relies on Python scientific librairies. The source code was tested in a Python 3.7 environnement. We provide a `requirements.txt` file to facilitate the installation of the necessary dependencies.

### Installation guide

Assuming the users have a working installation of Python on their computer (we recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/)), the users should create a new Python 3.7 environnement to avoid impacting on other file dependencies. The complete installation should be less than 15 minutes.

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
