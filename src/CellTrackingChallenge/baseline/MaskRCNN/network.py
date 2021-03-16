
import torch
import torchvision

def get_model_instance_segmentation(num_input_images, **kwargs):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(**kwargs)
    model.backbone.body.conv1 = torch.nn.Conv2d(
        num_input_images, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    return model
