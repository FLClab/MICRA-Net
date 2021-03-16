
import numpy
import torch

from torch import nn


class DoubleConvolver(nn.Module):
    """Class for the double convolution in the contracting path. The kernel size is
    set to 3x3 and no padding is inserted.

    :param in_channels: Number of channels in the input image
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Contracter(nn.Module):
    """Class for the max pooling operation followed by the contracting path.

    :param in_channels: Number of channels in the input image
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Contracter, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolver(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Expander(nn.Module):
    """Class for the expansion path of the UNet. An upsampling is performed and
    is followed by the convolution after adding the information of the contracting
    path.

    :param in_channels: Number of channels in the input image
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Expander, self).__init__()
        self.expand = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolver(in_channels=in_channels, out_channels=out_channels)

    def center_crop(self, bridge, target_size):
        _, _, bridge_height, bridge_width = bridge.size()
        diff_x = (bridge_height - target_size[0]) // 2
        diff_y = (bridge_width - target_size[1]) // 2
        return bridge[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x+target_size[1])]

    def forward(self, x, bridge):
        x = self.expand(x)
        crop = self.center_crop(bridge, x.size()[2 : ])
        concat = torch.cat([x, crop], 1)
        x = self.conv(concat)
        return x


class UNet(nn.Module):
    """Class for creating the UNet architecture. A first 2d convolution is done
    on the input image then the contracting path is created with a given depth and
    a set number of filter. The number of filter is doubled at every step.

    :param in_channels: Number of channels in the input image
    :param out_channels: Number of output channels from the UNet
    :param number_filter: Number of filters in the first layer (2 ** number_filter)
    :param depth: Depth of the network
    :param size: The size of the crops that are fed to the network
    """
    def __init__(self, in_channels, out_channels, number_filter=6, depth=5, **kwargs):
        super(UNet, self).__init__()
        self.outChannels = out_channels

        self.firstConvolution = DoubleConvolver(in_channels=in_channels, out_channels=2**number_filter)
        self.contractingPath = nn.ModuleList()
        for i in range(depth - 1):
            self.contractingPath.append(
                Contracter(in_channels=2**(number_filter + i), out_channels=2**(number_filter + i + 1))
            )
        self.expandingPath = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.expandingPath.append(
                Expander(in_channels=2**(number_filter + i + 1), out_channels=2**(number_filter + i))
            )
        self.lastConv = nn.Conv2d(in_channels=2**number_filter, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        knowledge = []
        x = self.firstConvolution(x)
        knowledge.append(x)
        for i, contracting in enumerate(self.contractingPath):
            x = contracting(x)
            if i != len(self.contractingPath) - 1:
                knowledge.append(x)
        for i, expanding in enumerate(self.expandingPath):
            x = expanding(x, knowledge[- i - 1])
        x = self.lastConv(x)
        x = torch.sigmoid(x)
        return x
