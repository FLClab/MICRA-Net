
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
from skimage import filters, morphology, transform, measure

def class_activation_map(network, X, cuda=False, size=256, num_classes=1):
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
                # local_map = (local_map - minimum[:, numpy.newaxis, numpy.newaxis]) / (maximum - minimum + 1e-12)[:, numpy.newaxis, numpy.newaxis]

                s = local_map.shape
                upsampled = local_map.repeat(size//s[1], axis=1).repeat(size//s[2], axis=2)
                # upsampled = transform.resize(local_map, (1, 1200, 1600))
                local_maps.append(upsampled)
            local_maps = numpy.swapaxes(numpy.array(local_maps), 0, 1)
            local_maps_class.append(local_maps.squeeze())
        CAM.append(numpy.array(local_maps_class))
        predictions.append(torch.sigmoid(pred).cpu().data.numpy())

    return numpy.array(CAM), numpy.array(predictions)

def segment_cam(cam, pred, X, size=256, num_classes=1):
    """
    Segments the class activation map of the network given the cam and the predictions

    :param cam: Class activation map of size (batch, class, layers, H, W)
    :param pred: Predictions of the network of size (batch, class)

    returns : The segmented map for each image in batch
    """
    pca = decomposition.PCA(n_components=1, whiten=False, svd_solver="randomized")
    segmented_precise, segmented_coarse, segmented_semantic = [], [], []
    unsegmented_precise = []
    for _cam, _pred, _X in zip(cam, pred, X):
        _X = _X.squeeze().cpu().data.numpy()
        total_precise = numpy.zeros((num_classes, *_cam.shape[-2:]))
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

            precise = filters.gaussian(transformed.reshape(size, size), sigma=3)
            raw_precise[p, :, :] = precise
            precise = precise > numpy.quantile(precise, q=0.8)
            total_precise[p, :, :] = precise.reshape(size, size)

        segmented_precise.append(total_precise)
        unsegmented_precise.append(raw_precise)
    return numpy.array(segmented_precise), numpy.array(unsegmented_precise)

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
