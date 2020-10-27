
import numpy
import torch
import os
import json

from tqdm import tqdm
from skimage import filters, morphology, measure

def class_activation_map(network, X, cuda=False, size=256, num_classes=1):

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
                # maximum[maximum == 0] = 1
                # local_map = (local_map - minimum[:, numpy.newaxis, numpy.newaxis]) / (maximum - minimum + 1e-12)[:, numpy.newaxis, numpy.newaxis]

                s = local_map.shape
                upsampled = local_map.repeat(size//s[1], axis=1).repeat(size//s[2], axis=2)
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
    segmented_precise = []
    unsegmented_precise = []
    for _cam, _pred, _X in zip(cam, pred, X):
        _X = _X.squeeze().cpu().data.numpy()
        total_precise = numpy.zeros((num_classes, *_cam.shape[-2:]))
        raw_precise = numpy.zeros((num_classes, *_cam.shape[-2:]))

        for p in numpy.argwhere(_pred == 1):
            local_maps = _cam[p].squeeze()
            reshaped = numpy.transpose(local_maps, axes=(1, 2, 0)).reshape(-1, local_maps.shape[0])

            index = (reshaped[:, 7] > 0)
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
            raw_precise[p] = filters.gaussian(transformed.reshape(size, size), sigma=1)
            total_precise[p] = threshold(raw_precise[p])

        segmented_precise.append(total_precise)
        unsegmented_precise.append(raw_precise)
    return numpy.array(segmented_precise), numpy.array(unsegmented_precise)

def threshold(raw):
    """
    Thresholds the raw image

    :param raw: A `numpy.ndarray` of the image

    :returns : A `numpy.ndarray` of the thresholded array
    """
    raw = numpy.pad(raw, ((0, 0), (1, 1), (1, 1)), mode="constant")
    raw = raw.squeeze()

    maximum_pred_pos = numpy.unravel_index(numpy.argmax(raw), raw.shape)

    precise = filters.gaussian(raw, sigma=5)
    precise = precise > numpy.quantile(precise, q=0.9)
    precise = morphology.remove_small_holes(precise, area_threshold=2500)

    precise_label_boundary, num_labels = measure.label(precise, return_num=True)
    precise_label = measure.label(precise)

    thresholded = numpy.zeros_like(precise)
    if num_labels < 1:
        return thresholded
    for j in range(1, num_labels + 1):
        sorted_vertices = measure.find_contours((precise_label == j).astype(float), 0.5, fully_connected="high")
        if sorted_vertices:
            for vertices in sorted_vertices:
                in_poly = measure.points_in_poly(numpy.array(maximum_pred_pos)[numpy.newaxis, ...], vertices)
                if in_poly:
                    thresholded = (precise_label == j).astype(int)[1:-1, 1:-1]
                    break
    return thresholded[numpy.newaxis, ...]
