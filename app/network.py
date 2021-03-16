
import numpy
import torch

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


def class_activation_map(network, X, cuda=False, size=256, num_classes=2):

    grad_eye = torch.eye(num_classes, requires_grad=True)
    if cuda:
        grad_eye = grad_eye.cuda()

    CAM, predictions = [], []
    for _X in X:
        pred = network.forward(_X.unsqueeze(0))
        local_maps_class = []
        for i in range(pred.shape[0]):
            network.zero_grad()
            pred.backward(grad_eye[i], retain_graph=True)
            local_maps = []
            for key in network.outputs.keys():

                A_k = network.outputs[key]
                grad = network.grads[key]

                alpha_k = grad.mean(axis=(2, 3))

                local_map = numpy.sum(A_k * alpha_k[:, :, numpy.newaxis, numpy.newaxis], axis=1)
                if key in ["4b"]:
                    local_map = numpy.maximum(local_map, 0)
                minimum = local_map.min(axis=(1, 2))
                maximum = local_map.max(axis=(1, 2))
                local_map = (local_map - minimum[:, numpy.newaxis, numpy.newaxis]) / (maximum - minimum + 1e-12)[:, numpy.newaxis, numpy.newaxis]

                s = local_map.shape
                upsampled = local_map.repeat(size//s[1], axis=1).repeat(size//s[2], axis=2)
                local_maps.append(upsampled)

            local_maps = numpy.swapaxes(numpy.array(local_maps), 0, 1)
            local_maps_class.append(local_maps.squeeze())
        CAM.append(numpy.array(local_maps_class))
        predictions.append((torch.sigmoid(pred) > 0.5).cpu().data.numpy())

    return numpy.array(CAM), numpy.array(predictions)
