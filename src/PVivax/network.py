
import numpy
import torch

from torch import nn

class MICRANet(nn.Module):
    """Class for creating the UNet architecture. A first 2d convolution is done
    on the input image then the contracting path is created with a given depth and
    a set number of filter. The number of filter is doubled at every step.

    :param in_channels: Number of channels in the input image
    :param out_channels: Number of output channels from the UNet
    :param number_filter: Number of filters in the first layer (2 ** number_filter)
    :param depth: Depth of the network
    :param size: The size of the crops that are fed to the network
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

    def forward(self, x):

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
        def hook(grad):
            self.grads[name] = grad.cpu().data.numpy()
        return hook
