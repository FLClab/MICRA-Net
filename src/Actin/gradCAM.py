
import numpy
import torch
import os
import json
import pickle

from tqdm import *
from torch.utils.data import DataLoader
from matplotlib import pyplot
from sklearn import cluster, decomposition, manifold
from skimage import filters, morphology

import utils

def class_activation_map(network, X, cuda=False, size=256):

    grad_eye = torch.eye(2, requires_grad=True)
    if cuda:
        grad_eye = grad_eye.cuda()

    CAM, predictions = [], []
    for _X in X:
        pred = network.forward(_X.unsqueeze(0))
        local_maps_class = []
        for i in range(pred.shape[0]):
            network.zero_grad()
            pred.backward(grad_eye[i], retain_graph=True)
            local_maps = []
            for key in network.outputs.keys():

                A_k = network.outputs[key]
                grad = network.grads[key]

                alpha_k = grad.mean(axis=(2, 3))

                local_map = numpy.sum(A_k * alpha_k[:, :, numpy.newaxis, numpy.newaxis], axis=1)
                if key in ["4b"]:
                    local_map = numpy.maximum(local_map, 0)
                minimum = local_map.min(axis=(1, 2))
                maximum = local_map.max(axis=(1, 2))
                local_map = (local_map - minimum[:, numpy.newaxis, numpy.newaxis]) / (maximum - minimum + 1e-12)[:, numpy.newaxis, numpy.newaxis]

                s = local_map.shape
                upsampled = local_map.repeat(size//s[1], axis=1).repeat(size//s[2], axis=2)
                local_maps.append(upsampled)

            local_maps = numpy.swapaxes(numpy.array(local_maps), 0, 1)
            local_maps_class.append(local_maps.squeeze())
        CAM.append(numpy.array(local_maps_class))
        predictions.append((torch.sigmoid(pred) > 0.5).cpu().data.numpy())

    return numpy.array(CAM), numpy.array(predictions)


def segment_cam(cam, pred, X, size=256):
    """
    Segments the class activation map of the network given the cam and the predictions

    :param cam: Class activation map of size (batch, class, layers, H, W)
    :param pred: Predictions of the network of size (batch, class)

    returns : The segmented map for each image in batch
    """
    pca = decomposition.PCA(n_components=1, whiten=False, svd_solver="randomized")
    segmented_precise = []
    unsegmented_precise = []
    for _cam, _pred, _X in zip(cam, pred, X):
        _X = _X.squeeze().cpu().data.numpy()
        total_precise = numpy.zeros((2, *_cam.shape[-2:]))
        raw_precise = numpy.zeros((2, *_cam.shape[-2:]))

        for p in numpy.argwhere(_pred == 1):
            local_maps = _cam[p].squeeze()
            reshaped = numpy.transpose(local_maps, axes=(1, 2, 0)).reshape(-1, local_maps.shape[0])

            index = (reshaped[:, 7] > 0) & utils.get_foreground(_X).ravel()
            if index.sum() < 1:
                continue

            repetitions = 5
            transformed = numpy.zeros((len(reshaped), 1))
            for _ in range(repetitions):
                transformed += pca.fit_transform(reshaped[:, :-1])
            transformed /= repetitions

            avg = numpy.average(transformed.ravel(), weights=reshaped[:, 7])

            m, M = transformed[index].min(), transformed[index].max()
            transformed -= m
            transformed /= (M - m)

            # Inverts the pca map
            if avg < 0:
                transformed -= 1
                transformed *= -1

            # Precise predictions
            precise = filters.gaussian(transformed.reshape(size, size), sigma=1)
            raw_precise[p, :, :] = precise
            precise = precise > numpy.quantile(precise, q=0.8)
            total_precise[p, :, :] = precise.reshape(size, size)

        segmented_precise.append(total_precise)
        unsegmented_precise.append(raw_precise)
    return numpy.array(segmented_precise), numpy.array(unsegmented_precise)
