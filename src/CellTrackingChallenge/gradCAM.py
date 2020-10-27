
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

def class_activation_map(network, X, cuda=False, size=256, num_classes=6):

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

    tot_precise = []
    unsegmented_precise = []
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
            precise[p] = threshold(raw_precise[p], p[0])

        unsegmented_precise.append(raw_precise)
        tot_precise.append(precise)
    return numpy.array(tot_precise), numpy.array(unsegmented_precise)

def threshold(raw, class_id):
    """
    Thresholds the raw image in a class specific manner

    :param raw: A `numpy.ndarray` of the image
    :param class_id: An `int` of the class id

    :returns : A `numpy.ndarray` of the thresholded array
    """
    class_names = [
        "DIC-C2DH-HeLa",
        "Fluo-C2DL-MSC",
        "Fluo-N2DH-GOWT1",
        "Fluo-N2DL-HeLa",
        "PhC-C2DH-U373",
        "PhC-C2DL-PSC",
    ]
    class_name = class_names[class_id]

    raw = raw.squeeze()

    if (class_name == "Fluo-N2DL-HeLa"):

        prediction = raw >= filters.threshold_otsu(raw[10:-10, 10:-10])

        prediction = morphology.remove_small_holes(prediction, area_threshold=100 * 100)
        prediction = morphology.binary_erosion(prediction, selem=morphology.disk(10))

        inverted_raw = 1 - raw
        inverted_raw[prediction == 0] = 0

        local_maximum = feature.peak_local_max(inverted_raw, min_distance=20,
                                                    threshold_rel=0.1,
                                                    indices=False, exclude_border=False)
        markers = ndimage.label(local_maximum)[0]
        prediction = segmentation.watershed(-inverted_raw, markers, mask=prediction, watershed_line=True)
        for _ in range(10): # Makes the watershed line rounder
            prediction = morphology.binary_erosion(prediction, selem=morphology.disk(4))
            prediction = morphology.binary_dilation(prediction, selem=morphology.disk(4))

    elif (class_name == "PhC-C2DL-PSC"):

        prediction = raw >= filters.threshold_otsu(raw[10:-10, 10:-10])

        prediction = morphology.remove_small_holes(prediction, area_threshold=100 * 100)
        prediction = morphology.binary_erosion(prediction, selem=morphology.disk(10))

        local_maximum = feature.peak_local_max(raw, min_distance=40,
                                                    threshold_rel=0.1,
                                                    indices=False, exclude_border=False)
        markers = ndimage.label(local_maximum)[0]
        prediction = segmentation.watershed(-raw, markers, mask=prediction, watershed_line=True)

        for _ in range(10): # Makes the watershed line rounder
            prediction = morphology.binary_erosion(prediction, selem=morphology.disk(4))
            prediction = morphology.binary_dilation(prediction, selem=morphology.disk(4))

    elif (class_name == "PhC-C2DH-U373"):

        prediction = raw >= filters.threshold_otsu(raw[10:-10, 10:-10])

        prediction = morphology.remove_small_holes(prediction, area_threshold=100 * 100)
        prediction = morphology.binary_erosion(prediction, selem=morphology.disk(15))

        prediction = morphology.remove_small_objects(prediction, min_size=25 * 25)

    elif (class_name == "Fluo-C2DL-MSC"):

        prediction = raw >= filters.threshold_triangle(raw[10:-10, 10:-10])

        prediction = ndimage.binary_fill_holes(prediction)
        prediction = morphology.remove_small_objects(prediction, min_size=50 * 50)

        distance = ndimage.distance_transform_edt(prediction)
        local_maximum = feature.peak_local_max(distance, min_distance=100,
                                                    indices=False, exclude_border=False)
        markers = ndimage.label(local_maximum)[0]
        prediction = segmentation.watershed(-distance, markers, mask=prediction, watershed_line=True)

    elif (class_name == "Fluo-N2DH-GOWT1"):

        prediction = raw >= filters.threshold_triangle(raw[10:-10, 10:-10])

        # coarse_segmentation = raw[1].astype(bool)
        prediction = morphology.remove_small_holes(prediction, area_threshold=100 * 100)
        prediction = morphology.binary_erosion(prediction, selem=morphology.disk(3))

        prediction = morphology.remove_small_objects(prediction, min_size=25 * 25)

        distance = ndimage.distance_transform_edt(prediction)
        local_maximum = feature.peak_local_max(distance, min_distance=25,
                                                    indices=False, exclude_border=False)
        markers = ndimage.label(local_maximum)[0]
        prediction = segmentation.watershed(-distance, markers, mask=prediction, watershed_line=True)

        prediction = morphology.binary_erosion(prediction, selem=morphology.disk(5))
        for _ in range(10): # Makes the watershed line rounder
            prediction = morphology.binary_erosion(prediction, selem=morphology.disk(4))
            prediction = morphology.binary_dilation(prediction, selem=morphology.disk(4))

    else:

        prediction = raw >= filters.threshold_otsu(raw[10:-10, 10:-10])

        prediction = morphology.remove_small_holes(prediction, area_threshold=100*100)
        prediction = numpy.logical_not(prediction)
        prediction = morphology.remove_small_holes(prediction, area_threshold=25 * 25)

        distance = ndimage.distance_transform_edt(prediction)
        local_maximum = feature.peak_local_max(distance, min_distance=100,
                                                indices=False, exclude_border=False)
        markers = ndimage.label(local_maximum)[0]
        prediction = segmentation.watershed(-distance, markers, mask=prediction, watershed_line=True)

    return prediction
