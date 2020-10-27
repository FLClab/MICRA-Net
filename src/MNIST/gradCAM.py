
import numpy
import torch
import os
import json
import pickle

from tqdm import *
from torch.utils.data import DataLoader
from matplotlib import pyplot
from sklearn import cluster, decomposition, manifold
from skimage import filters, feature, morphology, measure
from scipy import ndimage

def class_activation_map(network, X, cuda=False, size=256):
    """
    Creates the class activation maps for every layer of the network given an
    input image

    :param network: The network instance
    :param X: The data to evaluate
    :param cuda: Wheter to use cuda
    :param size: The size of the input images

    :returns : The class activation maps
               The prediction of the network
    """
    grad_eye = torch.eye(10, requires_grad=True)
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


def segment_cam(cam, pred, X):
    """
    Segments the class activation map of the network given the cam and the predictions

    :param cam: Class activation map of size (batch, class, layers, H, W)
    :param pred: Predictions of the network of size (batch, class)

    returns : The segmented map for each image in batch
    """
    pca = decomposition.PCA(n_components=1, whiten=False, svd_solver="randomized")
    segmented_precise, segmented_coarse = [], []
    for _cam, _pred, _X in zip(cam, pred, X):
        total_precise = numpy.zeros((10, *_cam.shape[-2:]))
        total_coarse = numpy.zeros((10, *_cam.shape[-2:]))

        for p in numpy.argwhere(_pred == 1):
            local_maps = _cam[p].squeeze()

            reshaped = numpy.transpose(local_maps, axes=(1, 2, 0)).reshape(-1, local_maps.shape[0])
            index = reshaped[:, 7] > 0

            reduced = numpy.zeros((len(reshaped), 1))
            repetitions = 5
            for _ in range(repetitions):
                reduced += pca.fit_transform(reshaped[:, :-1])
            reduced /= repetitions

            labels = measure.label(index.reshape(256, 256), connectivity=1)
            r_props = measure.regionprops(labels, intensity_image=reduced.reshape(256, 256))
            for region in r_props:
                img = region.intensity_image
                (min_row, min_col, max_row, max_col) = region.bbox
                certainty = local_maps[7, min_row : max_row, min_col : max_col].max()
                if certainty < 0.3:
                    continue
                thresholded = threshold_image(img)
                total_precise[p, min_row : max_row, min_col : max_col] = thresholded * (p + 1)
                total_coarse[p, min_row : max_row, min_col : max_col] = (local_maps[7, min_row : max_row, min_col : max_col] > 0).astype(int) * (p + 1)

        segmented_precise.append(total_precise)
        segmented_coarse.append(total_coarse)
    return numpy.array(segmented_precise), numpy.array(segmented_coarse)

def threshold_image(ary):
    """
    Thresholds an input array

    :param ary: A 2D numpy array

    :returns : A bunary 2D numpy array
    """
    reshaped_max = ary.reshape(ary.shape[0] // 8, 8, ary.shape[1] // 8, 8).max(-1).max(1)
    reshaped_min = ary.reshape(ary.shape[0] // 8, 8, ary.shape[1] // 8, 8).min(-1).min(1)
    delta_map = (reshaped_max - reshaped_min)
    delta_map = delta_map > filters.threshold_otsu(delta_map)
    temp = numpy.zeros(ary.shape)
    for j in range(delta_map.shape[0]):
        for i in range(delta_map.shape[1]):
            if delta_map[j, i]:
                thres = filters.threshold_otsu(ary[j * 8 : j * 8 + 8, i * 8 : i * 8 + 8])
                temp[j * 8 : j * 8 + 8, i * 8 : i * 8 + 8] = ary[j * 8 : j * 8 + 8, i * 8 : i * 8 + 8] > thres
    return temp
