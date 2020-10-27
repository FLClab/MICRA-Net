
import numpy
import torch
import os
import shutil
import time
import pickle
import json

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

import loader

from network import MICRANet

def _save_example(output_folder, epoch, X, y, pred):
    """
    Saves a matplotlib figure to the output folder as an example

    :param epoch: The epoch number
    :param X: The original image
    :param y: The labels
    :param pred: The prediction of the network
    """
    fig, ax = pyplot.subplots(figsize=(7,7))
    ax.imshow(X[0][0], cmap="gray", vmax=0.3*X.max())
    ax.set_title("Truth : {}\nPredicted : {}".format(y[0], pred[0]))
    pyplot.savefig(os.path.join(output_folder, "epoch_{}_example.pdf".format(epoch)), bbox_inches="tight")
    pyplot.close("all")

def calculate_accuracy(truth, pred):
    """
    Calculates the accuracy of the prediction

    :param truth: The true labels
    :param pred: The predicted labels
    """
    accurate = torch.sum(truth == (pred > 0).type(torch.float), dim=1)
    return (accurate == truth.shape[1]).type(torch.float).mean()

def save(output_folder, network, optimizer, stats):
    """
    Saves the network parameters and optimizer state in the given folder
    """
    torch.save(network.state_dict(), os.path.join(output_folder, "params.net"))
    torch.save(optimizer.state_dict(), os.path.join(output_folder, "optimizer.data"))
    pickle.dump(stats, open(os.path.join(output_folder, "statsCkpt.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    output_folder = os.path.join(".", "results", "model")
    try :
        os.makedirs(output_folder, exist_ok=True)
    except OSError as err:
        print("The name of the folder already exist! Try changing the name of the folder.")
        print("OSError", err)
        exit()

    data = numpy.load(os.path.join(".", "data", "mnist_train.npz"), allow_pickle=True)
    X, y = data["images"][:10], data["labels"][:10]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=True)

    lr, epochs, min_valid_loss = 0.01, 2, numpy.inf
    stats = defaultdict(list)
    trainer_params = {
        "in_channels" : 1,
        "num_classes" : 10,
        "number_filter" : 5,
        "depth" : 5,
        "size" : 256,
        "cuda" : False
    }
    json.dump(trainer_params, open(os.path.join(output_folder, "trainer_params.json"), "w"), indent=4, sort_keys=True)

    train_dataset = loader.Datasetter(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    valid_dataset = loader.Datasetter(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    network = MICRANet(**trainer_params)
    if trainer_params["cuda"]:
        network = network.cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr = lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()

    for epoch in range(epochs):

        start = time.time()
        print("Starting epoch {}/{}".format(epoch + 1, epochs))

        # Keep track of the loss of train and test
        statLossTrain, statLossTest = [], []

        # Puts the network in training mode
        network.train()
        train_accuracy = []
        for (X, y) in tqdm(train_loader, desc="Training"):

            # New batch we reset the optimizer
            optimizer.zero_grad()

            # Prediction and loss computation
            X = X.unsqueeze(1)
            if trainer_params["cuda"]:
                X = X.cuda()
                y = y.cuda()

            pred = network.forward(X)
            loss = criterion(pred, y)

            # Keeping track of statistics
            statLossTrain.append(loss.item())

            # Back-propagation and optimizer step
            loss.backward()
            optimizer.step()

            train_accuracy.append(calculate_accuracy(y, pred).item())

            # To avoid memory leak
            del X, y, pred, loss

        # Puts the network in evaluation mode
        network.eval()
        i, valid_accuracy = 0, []
        for (X, y) in tqdm(valid_loader , desc="Validation"):

            X = X.unsqueeze(1)
            if trainer_params["cuda"]:
                X = X.cuda()
                y = y.cuda()

            # Prediction and computation loss
            pred = network.forward(X)
            loss = criterion(pred, y)

            # Keeping track of statistics
            statLossTest.append(loss.item())

            if (i == 0):
                Xnumpy = X.cpu().data.numpy()
                ynumpy = y.cpu().data.numpy()
                prednumpy = pred.cpu().data.numpy()
                _save_example(output_folder, epoch, Xnumpy, ynumpy, prednumpy)

            valid_accuracy.append(calculate_accuracy(y, pred).item())

            # To avoid memory leaks
            del X, y, pred, loss
            i += 1

        # Aggregate stats
        for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                (numpy.mean, numpy.median, numpy.min, numpy.std)):
            stats[key].append(func(statLossTrain))
        for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                (numpy.mean, numpy.median, numpy.min, numpy.std)):
            stats[key].append(func(statLossTest))

        # Loss curves
        pyplot.figure(figsize=(10, 7))
        pyplot.plot(stats["trainMean"], linewidth=2, color="#2678B2", label="Train")
        pyplot.fill_between(numpy.arange(len(stats["trainMean"])),
                            numpy.array(stats["trainMean"]) - numpy.array(stats["trainStd"]),
                            numpy.array(stats["trainMean"]) + numpy.array(stats["trainStd"]),
                            color="#AFC8E7", alpha=0.7)
        pyplot.plot(stats["testMean"], linewidth=2, color="#FD7F28", label="Validation")
        pyplot.fill_between(numpy.arange(len(stats["testMean"])),
                            numpy.array(stats["testMean"]) - numpy.array(stats["trainStd"]),
                            numpy.array(stats["testMean"]) + numpy.array(stats["trainStd"]),
                            color="#FDBA7D", alpha=0.7)
        pyplot.legend()
        pyplot.xlabel("Epoch")
        pyplot.ylabel("Cross Entropy Loss over the predicted scores")
        pyplot.ylim(0, 1)
        pyplot.savefig(os.path.join(output_folder, "epoch_{}.pdf".format(epoch)), bbox_inches="tight")
        pyplot.close("all")

        isBest = False
        if min_valid_loss > stats["testMean"][-1]:
            print("New beat network ({} Cross Entropy is better than the previous {})".format(stats["testMean"][-1], min_valid_loss))
            min_valid_loss = stats["testMean"][-1]
            save(output_folder, network, optimizer, stats)
            isBest = True

        print("Epoch {} done!".format(epoch + 1))
        print("\tAvg loss train/validation : {:0.6f} / {:0.6f}".format(stats["trainMean"][-1], stats["testMean"][-1]))
        print("\tAvg accuracy train/validation : {:0.6f} / {:0.6f}".format(numpy.mean(train_accuracy), numpy.mean(valid_accuracy)))
        print("\tTook {} seconds".format(time.time() - start))
        pickle.dump(stats, open(os.path.join(output_folder, "stats.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
