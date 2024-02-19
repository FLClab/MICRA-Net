
import numpy
import os
import shutil
import time
import pickle
import json
import random
import datetime
import torch
import h5py

from torch import nn
from matplotlib import pyplot
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import loader
import network

def create_savefolder(params, dry_run=False):
    """
    Creates the savefolder and returns the path

    :param path: A `str` of path where to create the folder
    :param params: A `dict` of parameters

    :returns : A `str` of the created folder
    """
    output_folder = "{datetime}_{seed}".format(**params)
    if dry_run:
        output_folder = "dryrun"
    output_folder = os.path.join(params["savefolder"], output_folder)
    try :
        os.makedirs(output_folder, exist_ok=True)
    except OSError as err:
        print("The name of the folder already exist! Try changing the name of the folder.")
        print("OSError", err)
        exit()
    return output_folder

def print_params(trainer_params):
    """
    Prints trainer params

    :params trainer_params: A `dict` of the training parameters
    """
    def print_center(string, width):
        """
        Prints a string at the center of terminal
        """
        for _ in range(2) : print("-" * width)
        print(" " * (width // 2 - len(string) // 2) + string)
        for _ in range(2) : print("-" * width)

    def format_dict(d, previous_string=""):
        """
        Formats a `dict` to print
        """
        strings = []
        keys = sorted(list(d.keys()), key = lambda item : len(item))
        for key, values in d.items():
#             prev_spaces = " " * (len(keys[-1]) - len(key) + len(current_string))
            if previous_string:
                prev_spaces = " " * (len(keys[-1]) - len(key) + len(previous_string) - len(keys[-1]) - 4)
                prev_spaces = " " * (len(keys[-1]) - len(key) + len(previous_string) - len(keys[-1]) - 4)
            else:
                prev_spaces = " " * (len(keys[-1]) - len(key))
            string = f"{prev_spaces}{key}    :    "
            if isinstance(values, (dict)):
                string += "\n"
                values = format_dict(values, string)
            strings.append("".join((string, str(values))))
        return "\n".join(strings)

    def print_strings(strings, width):
        """
        Prints formated params at center window
        """
        keys = sorted(list(strings.split("\n")), key = lambda item : len(item))
        add_len = (width - len(keys[-1])) // 2
        for string in strings.split("\n"):
            print(" " * add_len + string)

    try:
        rows, columns = map(int, os.popen('stty size', 'r').read().split())
    except ValueError:
        rows, columns = 42, 100
    print_center("Training parameters", columns)
    strings = format_dict(trainer_params)
    print_strings(strings, columns)
    print_center("", columns)

def save_ckpt(output_folder, networks, trainer_params, model="MICRANet", filename="checkpoints.hdf5",
                verbose=True):
    """
    Saves the current network state to a hdf5 file. The architecture of the hdf5
    file is
    hdf5file
        MICRANet
            network

    :param output_folder: A `str` to the output folder
    :param networks: A `dict` of network models
    :param filename: (optional) A `str` of the filename. Defaults to "checkpoints.hdf5"
    :param verbose: (optional) Wheter the function in verbose
    """
    if verbose:
        print("[----]     Saving current network state")
    with h5py.File(os.path.join(output_folder, filename), "a") as file:

        if model in file:
            del file[model]

        model_group = file.create_group(model)
        model_group.attrs["trainer_params"] = json.dumps(trainer_params, sort_keys=True, indent=4)
        for network_name, params in networks.items():
            group = model_group.create_group(network_name)
            for key, values in params.state_dict().items():
                group.create_dataset(key, data=values.cpu().data.numpy())

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Performs a dry run")
    parser.add_argument("-s", "--size", type=str, default="256",
                        help="The size of crops to use.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sets the default random seed")
    parser.add_argument("--num", type=int, default=1,
                        help="Sets the number of repetitions")
    parser.add_argument("--cuda", action="store_true",
                        help="(optional) activate cuda")
    parser.add_argument("--restore-from", type=str, default=None,
                        help="(optional) activate cuda")    
    parser.add_argument("--freeze", action="store_true",
                        help="(optional) freeze convolutional layers")    
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="(optional) uses tensorboard for logging")
    args = parser.parse_args()

    PATH = "./MICRA-Net"

    add_to_seed = -1
    for _ in range(args.num):

        add_to_seed += 1

        torch.manual_seed(args.seed + add_to_seed)
        torch.cuda.manual_seed(args.seed + add_to_seed)
        numpy.random.seed(args.seed + add_to_seed)
        random.seed(args.seed + add_to_seed)
        torch.backends.cudnn.enabled=False
        torch.backends.cudnn.deterministic=True

        hdf5_training_path = f"{PATH}/datasets/training_01-04-19.hdf5"
        hdf5_validation_path = f"{PATH}/datasets/validation_01-04-19.hdf5"
        hdf5_testing_path = f"{PATH}/datasets/training_01-04-19.hdf5"

        lr, epochs, min_valid_loss = 1e-3, 250 if not args.dry_run else 1, numpy.inf
        pos_weight = [3.3, 1.6]
        stats = defaultdict(list)
        current_datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
        trainer_params = {
            "model_name" : "_".join((current_datetime, str(args.seed + add_to_seed))),
            "restore-from" : args.restore_from,
            "freeze" : args.freeze,
            "savefolder" : f"{PATH}/Results",
            "datetime" : current_datetime,
            "dry_run" : args.dry_run,
            "size" : eval(args.size),
            "seed" : args.seed + add_to_seed,
            "lr" : lr,
            "epochs" : epochs,
            "cuda" : args.cuda and torch.cuda.is_available(),
            "data_aug" : 0.5,
            "step" : 0.75,
            "pos_weight" : pos_weight,
            "hdf5_training_path" : hdf5_training_path,
            "hdf5_validation_path" : hdf5_validation_path,
            "hdf5_testing_path" : hdf5_testing_path,
            "dataloader_params" : {
                "shuffle" : True,
                "num_workers" : 4,
                "pin_memory" : True,
                "drop_last" : True,
                "batch_size" : 64,
            },
            "model_params" : {
                "num_classes" : 2,
                "num_input_images" : 1,
            },
            "scheduler" : {
                "patience" : 10,
                "threshold" : 0.01,
                "min_lr" : 1e-5,
                "factor" : 0.1,
                "verbose" : True
            }
        }
        print_params(trainer_params)

        # Creation of saving folder
        output_folder = create_savefolder(trainer_params, dry_run=trainer_params["dry_run"])
        json.dump(trainer_params, open(os.path.join(output_folder, "trainer_params.json"), "w"), indent=4, sort_keys=True)

        # Tensorboard logging
        if args.use_tensorboard:
            writer = SummaryWriter(os.path.join(output_folder, "logs"))

        # Creation of the loaders
        train_dataset = loader.HDF5Dataset(trainer_params["hdf5_training_path"], **trainer_params)
        train_loader = DataLoader(train_dataset, **trainer_params["dataloader_params"])
        valid_dataset = loader.HDF5Dataset(trainer_params["hdf5_validation_path"], validation=True, **trainer_params)
        valid_loader = DataLoader(valid_dataset, **trainer_params["dataloader_params"])

        model = network.MICRANet(**trainer_params, **trainer_params["model_params"])
        if trainer_params["restore-from"]:
            model.restore_from(trainer_params["restore-from"], freeze_conv_layers=trainer_params["freeze"])
        if trainer_params["cuda"]:
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(trainer_params["pos_weight"]))
        if trainer_params["cuda"]:
            criterion = criterion.cuda()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **trainer_params["scheduler"])
        for epoch in range(epochs if args.dry_run else epochs):

            start = time.time()
            print("[----] Starting epoch {}/{}".format(epoch + 1, epochs))

            # Keep track of the loss of train and test
            statLossTrain, statLossTest = defaultdict(list), defaultdict(list)

            # Puts the model in training mode
            model.train()
            for i, (X, y) in enumerate(tqdm(train_loader, desc="[----] ")):

                # Reshape
                if X.dim() == 3:
                    X = X.unsqueeze(1)

                # Send to gpu
                if trainer_params["cuda"]:
                    X = X.cuda()
                    y = y.cuda()
                y = y.squeeze()

                # Prediction and loss computation
                pred = model.forward(X)
                loss = criterion(pred, y)

                # Keeping track of statistics
                statLossTrain["loss"].append(loss.item())
                statLossTrain["accuracy"].append(numpy.sum((y.cpu().data.numpy()>0).astype(int) * (pred.cpu().data.numpy()>=0.5).astype(int), axis=0) / len(y))

                # Back-propagation and optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # To avoid memory leak
                torch.cuda.empty_cache()
                del X, y, loss

            # Puts the model in evaluation mode
            model.eval()
            for i, (X, y) in enumerate(tqdm(valid_loader, desc="[----] ")):

                # Reshape
                if X.dim() == 3:
                    X = X.unsqueeze(1)
                y = y.squeeze()

                # Send to gpu
                if trainer_params["cuda"]:
                    X = X.cuda()
                    y = y.cuda()

                # Prediction and loss computation
                pred = model.forward(X)
                loss = criterion(pred, y)

                # Keeping track of statistics
                statLossTest["loss"].append(loss.item())
                statLossTest["accuracy"].append(numpy.sum((y.cpu().data.numpy()>0).astype(int) * (pred.cpu().data.numpy()>=0.5).astype(int), axis=0) / len(y))

                # To avoid memory leak
                torch.cuda.empty_cache()
                del X, y, loss

            # Aggregate stats
            for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                    (numpy.mean, numpy.median, numpy.min, numpy.std)):
                stats[key].append(func(statLossTrain["loss"]))
                if args.use_tensorboard:
                    writer.add_scalar(f"Loss/{key}", stats[key][-1], epoch)
            for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                    (numpy.mean, numpy.median, numpy.min, numpy.std)):
                stats[key].append(func(statLossTest["loss"]))
                if args.use_tensorboard:
                    writer.add_scalar(f"Loss/{key}", stats[key][-1], epoch)
            for key, func in zip(("trainAccMean", "trainAccMed", "trainAccMin", "trainAccStd"),
                    (numpy.mean, numpy.median, numpy.min, numpy.std)):
                stats[key].append(func(statLossTrain["accuracy"]))
                if args.use_tensorboard:
                    writer.add_scalar(f"Acc/{key}", stats[key][-1], epoch)
            for key, func in zip(("testAccMean", "testAccMed", "testAccMin", "testAccStd"),
                    (numpy.mean, numpy.median, numpy.min, numpy.std)):
                stats[key].append(func(statLossTest["accuracy"]))
                if args.use_tensorboard:
                    writer.add_scalar(f"Acc/{key}", stats[key][-1], epoch)      
            stats["lr"].append(optimizer.param_groups[0]["lr"])
            if args.use_tensorboard:
                writer.add_scalar(f"lr", stats["lr"][-1], epoch)
            scheduler.step(numpy.min(stats["testMean"]))

            # Save if best model so far
            if min_valid_loss > stats["testMean"][-1]:
                print("[!!!!] New beat model ({:0.4f}. Loss is better than the previous {:0.4f})".format(stats["testMean"][-1], min_valid_loss))
                min_valid_loss = stats["testMean"][-1]
                networks = {
                    trainer_params["model_name"] : model
                }
                save_ckpt(output_folder, networks=networks, trainer_params=trainer_params)

                del networks

            print("[----] Epoch {} done!".format(epoch + 1))
            print("[----]     Avg loss train/validation : {:0.4f} / {:0.4f}".format(stats["trainMean"][-1], stats["testMean"][-1]))
            print("[----]     Avg accuracy train/validation : {:0.4f} / {:0.4f}".format(stats["trainAccMean"][-1], stats["testAccMean"][-1]))
            print("[----]     Current best model : {:0.4f}".format(min_valid_loss))
            print("[----]     Current learning rate : {:0.4e}".format(optimizer.param_groups[0]["lr"]))
            print("[----]     Took {} seconds".format(time.time() - start))
            pickle.dump(stats, open(os.path.join(output_folder, "stats.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # Avoids memory leaks
        del train_loader, valid_loader, model, optimizer
