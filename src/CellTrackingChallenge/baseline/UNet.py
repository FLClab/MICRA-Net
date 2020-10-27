
import numpy
import h5py
import torch

from torch import nn

class UNet(nn.Module):
    """
    Implements the UNet architecture from~\cite{falk2019u}.
    Contrary to the seminal implementation of UNet, we use padding to output a
    predicted tensor with the same shape

    @article{falk2019u,
      title={U-Net: deep learning for cell counting, detection, and morphometry},
      author={Falk, Thorsten and Mai, Dominic and Bensch, Robert and {\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Marrakchi, Yassine and B{\"o}hm, Anton and Deubner, Jan and J{\"a}ckel, Zoe and Seiwald, Katharina and others},
      journal={Nature methods},
      volume={16},
      number={1},
      pages={67},
      year={2019},
      publisher={Nature Publishing Group}
    }
    """
    def __init__(self, **kwargs):
        """
        Instantiates the `UNet` class
        """
        super(UNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

        self.conv_d0ab = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv_d0bc = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv_d1ab = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_d1bc = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv_d2ab = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_d2bc = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv_d3ab = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_d3bc = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv_d4ab = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv_d4bc = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.upconv_d4c_u3a = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.conv_u3bc = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv_d3cd = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.upconv_u3d_u2a = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.conv_u2bc = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_d2cd = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.upconv_u2d_u1a = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.conv_u1bc = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_d1cd = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.upconv_u1d_u0a = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.conv_u0bc = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1)
        self.conv_d0cd = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv_u0dscore = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1)

    def double_conv(self, x, layers):
        """
        Implements the double convolutions

        :param x: A `torch.Tensor` to convolve
        :param layers: A `list` of convolution to apply to x

        :returns : A `torch.Tensor`
        """
        for layer in layers:
            x = self.leakyrelu(layer(x))
        return x

    def forward(self, x):
        """
        Implements the forward method of `nn.Module`

        :param x: A `torch.Tensor` with shape [B, 1, H, W]

        :returns : A `torch.Tensor` with shape [B, 2, H, W]
        """

        x = self.leakyrelu(self.conv_d0ab(x))
        x1 = self.leakyrelu(self.conv_d0bc(x))

        x = self.maxpool(x1)
        x2 = self.double_conv(x, [self.conv_d1ab, self.conv_d1bc])

        x = self.maxpool(x2)
        x3 = self.double_conv(x, [self.conv_d2ab, self.conv_d2bc])

        x = self.maxpool(x3)
        x4 = self.double_conv(x, [self.conv_d3ab, self.conv_d3bc])

        x = self.maxpool(x4)
        x = self.double_conv(x, [self.conv_d4ab, self.conv_d4bc])

        x = self.upconv_d4c_u3a(x)
        x = torch.cat([x4, x], dim=1)
        x = self.double_conv(x, [self.conv_u3bc, self.conv_d3cd])

        x = self.upconv_u3d_u2a(x)
        x = torch.cat([x3, x], dim=1)
        x = self.double_conv(x, [self.conv_u2bc, self.conv_d2cd])

        x = self.upconv_u2d_u1a(x)
        x = torch.cat([x2, x], dim=1)
        x = self.double_conv(x, [self.conv_u1bc, self.conv_d1cd])

        x = self.upconv_u1d_u0a(x)
        x = torch.cat([x1, x], dim=1)
        x = self.double_conv(x, [self.conv_u0bc, self.conv_d0cd])

        x = self.conv_u0dscore(x)

        return x
