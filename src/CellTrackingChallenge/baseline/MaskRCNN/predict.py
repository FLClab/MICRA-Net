
import numpy
import torch
import os
import glob
import json
import h5py

from tqdm import tqdm
from skimage import io
from matplotlib import pyplot
from torch.utils.data import DataLoader

import network
import loader

def get_masks(boxes=None, scores=None, masks=None, labels=None, ax=None, num_classes=None, **kwargs):
    """
    Plots the boxes on the ax

    :param boxes: A `torch.tensor` of shape [N, 4] of the boxes
    :param scores: A `torch.tensor` of scores of each boxes
    :param masks: A `torch.tensor` of predicted masks of each boxes
    :param labels: A `torch.tensor` of predicted labels of each boxes
    :param ax: A `matplotlib.Axes` where to plot the predicted boxes
    """

    cmap = pyplot.get_cmap("hot")

    boxes = boxes.cpu().data.numpy()
    scores = scores.cpu().data.numpy()
    masks = masks.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    cells = numpy.zeros((num_classes, *masks.shape[-2:]))
    objs = numpy.zeros((1, *masks.shape[-2:]))

    nms_thres = 0.7
    if "nms_thres" in kwargs:
        nms_thres = kwargs["nms_thres"]
    if "score_thres" in kwargs:
        score_thres = kwargs["score_thres"]
        boxes, scores, masks, labels = boxes[scores >= score_thres], scores[scores >= score_thres], masks[scores >= score_thres], labels[scores >= score_thres]

    if len(boxes) > 1:
        picks = non_max_suppression(boxes, scores, nms_thres)
    elif len(boxes) == 1:
        picks = [0]
    else:
        picks = []
    picks = picks[::-1]
    for i, (box, score, mask, label) in enumerate(zip(boxes[picks], scores[picks], masks[picks], labels[picks])):

        box = box.astype(int)
#         if "pad_size" in kwargs:
#             pad_size = kwargs["pad_size"]
#             if any(box >= mask.shape[-1] - pad_size) or any(box <= pad_size):
#                 continue

        slc = (label, slice(box[1], box[3]), slice(box[0], box[2]))
        binary_mask = numpy.zeros(cells.shape)
        binary_mask[label] = mask[0] >= 0.5 # Threshold from mask rcnn paper
        binary_mask = binary_mask.astype(bool)
        cells[binary_mask] = 1

        mask = mask >= 0.5
        objs[mask] = (i + 1)

        if "pad_size" in kwargs:
            pad_size = kwargs["pad_size"]
            cells = cells[:, pad_size:-pad_size, pad_size:-pad_size]
            cells = numpy.pad(cells, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode="edge")

            objs = objs[:, pad_size:-pad_size, pad_size:-pad_size]
            objs = numpy.pad(objs, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode="edge")

        if not isinstance(ax, type(None)):
            rect = patches.Rectangle(
                xy=(box[0], box[1]),
                width = box[2] - box[0], height = box[3] - box[1],
                facecolor="none", edgecolor=cmap(score), linestyle="dashed"
            )
            ax.add_artist(rect)

    return cells, objs

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = numpy.maximum(box[0], boxes[:, 0])
    x2 = numpy.minimum(box[2], boxes[:, 2])
    y1 = numpy.maximum(box[1], boxes[:, 1])
    y2 = numpy.minimum(box[3], boxes[:, 3])

    intersection = numpy.maximum(x2 - x1, 0) * numpy.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes:
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(numpy.float32)

    # Compute box areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = numpy.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = numpy.delete(ixs, remove_ixs)
        ixs = numpy.delete(ixs, 0)
    return numpy.array(pick, dtype=numpy.int32)

class Predicter:
    """
    Class to make predictions on a set of images
    """
    def __init__(self, data_path, model_path, save_folder, cuda=False, supervision="FS"):
        """
        Inits the predicter class

        :param data_path: A file path to the dataset
        :param model_path: A file path to the model
        :param save_folder: A file path where to save the files
        :param cuda: Wheter to use cuda
        """
        # Assigns member variables
        self.data_path = data_path
        self.model_path = model_path
        self.save_folder = save_folder
        self.cuda = cuda
        self.supervision = supervision

        # Loads the model
        self.load_network()

        self.image_folder_iterator = loader.ImageFolderIterator(self.data_path)

    def predict(self):
        """
        Predicts the data from the dataloader. Saves the output save_folder
        """
        size = self.trainer_params["size"]
        num_classes = self.trainer_params["model_params"]["num_classes"]

        for i, image in enumerate(tqdm(self.image_folder_iterator)):

            # Pads the image to avoid undesired edge effect
            if image.ndim == 2:
                image = numpy.expand_dims(image, 0)
            image = numpy.pad(
                image, ((0, 0), (size // 2, size // 2), (size // 2, size // 2)), mode="symmetric"
            )

            # Creates loader of the images
            dataset = loader.Datasetter(image, self.trainer_params)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

            # Creates the prediction builder
            pb = loader.PredictionBuilder(image.shape[-2:], size, num_classes=num_classes)
            for X, (positions_y, positions_x) in dataloader:
                if X.ndim == 3:
                    X = X.unsqueeze(1)

                if self.cuda:
                    X = X.cuda()

                pred = self.model.forward(X)
                masks = []
                for p in pred:
                    mask, objs = get_masks(**p, score_thres=0.5, num_classes=num_classes)
                    masks.append(mask)
                masks = numpy.array(masks)

                for pos_y, pos_x, mask in zip(positions_y, positions_x, masks):
                    pb.add_predictions_ji(mask, pos_y, pos_x)

            # Return current prediction and crops the padded area
            masks = pb.return_prediction()
            masks = masks[:, size // 2 : -size // 2,
                             size // 2 : -size // 2]
            self.save_images(i, X.cpu().data.numpy(), masks)

    def load_network(self):
        """
        Loads the model from model_path
        """
        net_params, self.trainer_params = self.load()
        self.model = network.get_model_instance_segmentation(**self.trainer_params["model_params"])
        self.model.eval()
        self.model.load_state_dict(net_params)
        if self.cuda:
            self.model = self.model.cuda()

    def save_images(self, batch, images, pred):
        """
        Saves the images, masks, local maps, precise, coarse and predictions
        """
        io.imsave(os.path.join(self.save_folder, "{}_image.tif".format(batch)), images.astype(numpy.float32), check_contrast=False)
        io.imsave(os.path.join(self.save_folder, "{}_pred.tif".format(batch)), pred.astype(numpy.float32), check_contrast=False)

    def load(self):
        """
        Loads a previous network and optimizer state
        """
        with h5py.File(self.model_path, "r") as file:
            networks = {}
            for key, values in file["-".join(("MaskRCNN", self.supervision))].items():
                networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
            trainer_params = json.loads(file["-".join(("MaskRCNN", self.supervision))].attrs["trainer_params"])
        net_params = networks[key]
        return net_params, trainer_params

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="(optional) Wheter cuda can be used")
    parser.add_argument("--supervision", type=str, default="FS",
                        help="(optional) Which supervision level to load")
    args = parser.parse_args()

    available_supervision = ["FS", "ALT2", "ALT5", "ALT10", "BBOX"]
    assert args.supervision in available_supervision, "The supervision level does not exists... Here's the valid list : [{}]".format(", ".join(available_supervision))

    data_path = os.path.join("..", "..", "data")
    model_path = os.path.join("..", "..", "MICRA-Net", "models", "CTCModelZoo.hdf5")
    save_folder = os.path.join(".", "segmentation")
    os.makedirs(save_folder, exist_ok=True)

    predicter = Predicter(data_path, model_path, save_folder, cuda=args.cuda, supervision=args.supervision)
    predicter.predict()
