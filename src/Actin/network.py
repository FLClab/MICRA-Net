
import numpy
import torch
import os
import h5py
import json

from torch import nn

class MICRANet(nn.Module):
    """
    Class for creating the `MICRANet` architecture

    :param grad: (optional) Wheter the gradient should be calculated
    """
    def __init__(self, grad=False, **kwargs):
        super(MICRANet, self).__init__()
        self.grad = grad

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1a = nn.Conv2d(in_channels=kwargs["num_input_images"], out_channels=32, kernel_size=3, padding=1)
        self.bnorm1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1b = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bnorm2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bnorm2b = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bnorm3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bnorm3b = nn.BatchNorm2d(128)

        self.conv4a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bnorm4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm4b = nn.BatchNorm2d(256)

        self.global_pool = nn.MaxPool2d(kernel_size=kwargs["size"] // 2 ** 3)
        self.linear1 = nn.Linear(256, kwargs["num_classes"])
        self.dropout = nn.Dropout(p=0.5)

        self.grads = {}
        self.outputs = {}

    def restore_from(self, path, freeze_conv_layers=False):
        """
        Restores model weights from path

        :param path: A `str` of the path
        """
        ext = os.path.splitext(path)[-1]
        if ext in [".h5", ".hfd5"]:
            with h5py.File(path, "r") as file:
                networks = {}
                for key, values in file["MICRANet"].items():
                    networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
                trainer_params = json.loads(file["MICRANet"].attrs["trainer_params"])
            net_params = networks[key]
            super().load_state_dict(net_params, strict=False)
        elif ext in [".pt", ".pth"]:
            ckpt = torch.load(path, map_location="cpu")
            net_params = ckpt["model"]["backbone"]
            super().load_state_dict(net_params, strict=False)
        else:
            raise NotImplementedError(f"`{ext}` is not a valid checkpoint.")

        if freeze_conv_layers:
            for key, param in self.named_parameters():
                if not "linear1" in key:
                    param.requires_grad = False

    def forward(self, x):
        """
        Implements the forward method of `MICRANet`

        :param x: A `torch.tensor` of the input data

        :returns : A `torch.tensor` of the classified input data
        """
        x = nn.functional.relu(self.bnorm1a(self.conv1a(x)))
        if self.grad:
            x.register_hook(self.save_grad("1a"))
            self.outputs["1a"] = x.clone().detach().cpu().data.numpy()
        x = nn.functional.relu(self.bnorm1b(self.conv1b(x)))
        if self.grad:
            x.register_hook(self.save_grad("1b"))
            self.outputs["1b"] = x.clone().detach().cpu().data.numpy()
        x = self.maxpool(x)

        # 64 x 64
        x = nn.functional.relu(self.bnorm2a(self.conv2a(x)))
        if self.grad:
            x.register_hook(self.save_grad("2a"))
            self.outputs["2a"] = x.clone().detach().cpu().data.numpy()
        x = nn.functional.relu(self.bnorm2b(self.conv2b(x)))
        if self.grad:
            x.register_hook(self.save_grad("2b"))
            self.outputs["2b"] = x.clone().detach().cpu().data.numpy()
        x = self.maxpool(x)

        # 32 x 32
        x = nn.functional.relu(self.bnorm3a(self.conv3a(x)))
        if self.grad:
            x.register_hook(self.save_grad("3a"))
            self.outputs["3a"] = x.clone().detach().cpu().data.numpy()
        x = nn.functional.relu(self.bnorm3b(self.conv3b(x)))
        if self.grad:
            x.register_hook(self.save_grad("3b"))
            self.outputs["3b"] = x.clone().detach().cpu().data.numpy()
        x = self.maxpool(x)

        # 16 x 16
        x = nn.functional.relu(self.bnorm4a(self.conv4a(x)))
        if self.grad:
            x.register_hook(self.save_grad("4a"))
            self.outputs["4a"] = x.clone().detach().cpu().data.numpy()
        x = nn.functional.relu(self.bnorm4b(self.conv4b(x)))
        if self.grad:
            x.register_hook(self.save_grad("4b"))
            self.outputs["4b"] = x.clone().detach().cpu().data.numpy()

        x = self.global_pool(x).squeeze()
        x = self.linear1(self.dropout(x))
        return x

    def save_grad(self, name):
        """
        Implements a storing method of the gradients

        :param name: A `str` of the name of the layer
        """
        def hook(grad):
            self.grads[name] = grad.cpu().data.numpy()
        return hook
