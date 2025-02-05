import torchvision.models as models
from ..utils.convert_activation import convert_relu_to_softplus
import torch.nn as nn


def load_model(model, activation_fn, softplus_beta, freeze_bn=False):
    # num_classes is used after initialize model.
    if model == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model == "convnext_t":
        net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif model == "swin_v2_t":
        net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
    else:
        raise NameError(f"{model} is a wrong model")

    if activation_fn == "softplus":
        convert_relu_to_softplus(net, softplus_beta)

    if freeze_bn:
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):  # or BatchNorm1d, BatchNorm3d
                module.track_running_stats = False

    return net
