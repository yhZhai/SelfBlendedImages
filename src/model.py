import torch
from torch import nn
import timm
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM


class Detector(nn.Module):
    def __init__(self, args, num_classes=2):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained(
            "efficientnet-b4", advprop=True, num_classes=num_classes
        )
        # self.net = timm.create_model("twins_svt_large", pretrained=True)
        # self.net.head = nn.Linear(1024, 2, bias=True)
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(
            self.parameters(), torch.optim.SGD, lr=args.lr, momentum=args.momentum
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, x, target):
        for i in range(2):
            pred_cls = self(x)
            if i == 0:
                pred_first = pred_cls
            loss_cls = self.cel(pred_cls, target)
            loss = loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return pred_first


if __name__ == "__main__":
    model = timm.create_model("eva_large_patch14_336.in22k_ft_in22k_in1k", pretrained=True, num_classes=7)
    print(model)
