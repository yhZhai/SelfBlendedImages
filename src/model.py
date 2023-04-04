import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained(
            "efficientnet-b4", advprop=True, num_classes=2
        )
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, x, target):
        self.optimizer.zero_grad()
        pred_cls = self(x)
        loss_cls = self.cel(pred_cls, target)
        loss_cls.backward()
        self.optimizer.step()

        return pred_cls
