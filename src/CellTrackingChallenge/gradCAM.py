
import numpy
import torch
import os
import json
import pickle
# import umap

from tqdm import *
from torch.utils.data import DataLoader
from matplotlib import pyplot
from sklearn import cluster, decomposition, manifold
from skimage import filters, morphology, transform, feature, segmentation
from scipy import ndimage

def class_activation_map(network, X, cuda=False, size=256, num_classes=10):
    """
    Extracts the grad-CAMs from the network

    :param network: A network model
    :param X: A `torch.tensor` of the images
    :param cuda: (optional) Wheter the calculation should be done on GPU
    :param size: (optional) An `int` of the size
    :param num_classes: (optional) An `int` of the number of classes

    :returns : A `numpy.ndarray` of local maps with shape (batch, 10, num_classes, size, size)
               A `numpy.ndarray` of the prediction of the model
    """
    grad_eye = torch.eye(num_classes, requires_grad=True)
    if cuda:
        grad_eye = grad_eye.cuda()

    CAM, predictions = [], []
    # for _X in tqdm(X, leave=False):
    for _X in X:
        pred = network.forward(_X.unsqueeze(0))
        local_maps_class = []
        for i in range(num_classes):
            network.zero_grad()
            pred.backward(grad_eye[i], retain_graph=True)
            local_maps = []
            for key in sorted(network.outputs.keys()):

                A_k = network.outputs[key]
                grad = network.grads[key]

                alpha_k = grad.mean(axis=(2, 3))

                local_map = numpy.sum(A_k * alpha_k[:, :, numpy.newaxis, numpy.newaxis], axis=1)
                if key in ["4b"]:
                    local_map = numpy.maximum(local_map, 0)
                minimum = local_map.min(axis=(1, 2))
                maximum = local_map.max(axis=(1, 2))
                # maximum[maximum == 0] = 1
                local_map = (local_map - minimum[:, numpy.newaxis, numpy.newaxis]) / (maximum - minimum + 1e-12)[:, numpy.newaxis, numpy.newaxis]

                s = local_map.shape
                upsampled = local_map.repeat(size//s[1], axis=1).repeat(size//s[2], axis=2)
                # upsampled = transform.resize(local_map, (1, 1200, 1600))
                local_maps.append(upsampled)

            local_maps = numpy.swapaxes(numpy.array(local_maps), 0, 1)
            local_maps_class.append(local_maps.squeeze())
        CAM.append(numpy.array(local_maps_class))
        predictions.append((torch.sigmoid(pred) > 0.5).cpu().data.numpy())
    return numpy.array(CAM), numpy.array(predictions)

def segment_cam(cam, pred, X, size=256, num_classes=6):
    """
    Segments the class activation map of the network given the cam and the predictions

    :param cam: Class activation map of size (batch, class, layers, H, W)
    :param pred: Predictions of the network of size (batch, class)

    returns : The segmented map for each image in batch
    """
    pca = decomposition.PCA(n_components=1, whiten=False, svd_solver="randomized")

    masks, raw_predictions = [], []
    for _cam, _pred, _X in zip(cam, pred, X):
        _X = _X.squeeze().cpu().data.numpy()

        precise = numpy.zeros((num_classes, *_cam.shape[-2:]))
        raw_precise = numpy.zeros((num_classes, *_cam.shape[-2:]))

        for p in numpy.argwhere(_pred == 1):
            local_maps = _cam[p].squeeze()
            reshaped = numpy.transpose(local_maps, axes=(1, 2, 0)).reshape(-1, local_maps.shape[0])

            index = (reshaped[:, 7] > 0)

            repetitions = 5
            transformed = numpy.zeros((len(reshaped), 1))
            for _ in range(repetitions):
                transformed += pca.fit_transform(reshaped[:, :-1])
            transformed /= repetitions

            m, M = transformed[index].min(), transformed[index].max()
            transformed -= m
            transformed /= (M - m)

            raw_precise[p, :, :] = filters.gaussian(transformed.reshape(*_cam.shape[-2:]), sigma=3)

        masks.append(threshold(raw_precise))
        raw_predictions.append(raw_precise)
    return numpy.array(masks), numpy.array(raw_predictions)

def threshold(raw):
    """
    Thresholds the raw prediction from the network model

    :param raw: A `numpy.ndarray` of the image

    :returns : A `numpy.ndarray` of the thresholded array
    """
    diff = raw[::2] - raw[1::2]
    mask = numpy.zeros_like(diff)
    for i, per_class_diff in enumerate(diff):
        if numpy.any(per_class_diff):
            threshold = filters.threshold_otsu(per_class_diff[per_class_diff != 0])
        else:
            threshold = 1
        mask[i] = per_class_diff >= threshold
    return mask
