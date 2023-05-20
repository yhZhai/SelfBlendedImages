import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from einops import rearrange
from efficientnet_pytorch import EfficientNet


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained(
            "efficientnet-b4", advprop=True, num_classes=2
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def get_cam(self, x):
        feature = self.net.extract_features(x)
        feature = rearrange(feature, "b c h w -> b h w c")
        cam = self.net._fc(feature)
        return cam
