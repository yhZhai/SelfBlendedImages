from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights

from datasets.flow_dataset import FlowDataset
from model import Detector
from opt import get_args
from train_sbi import compute_accuray
from utils.misc import MetricLogger, setup_env


def main(args):
    writer = setup_env(args)

    train_dataset = FlowDataset(
        "train",
        ["data/FaceForensics++/original_sequences/youtube/c23/flow"],
        [
            "data/FaceForensics++/manipulated_sequences/Deepfakes/c23/flow",
            "data/FaceForensics++/manipulated_sequences/Face2Face/c23/flow",
            "data/FaceForensics++/manipulated_sequences/FaceSwap/c23/flow",
            "data/FaceForensics++/manipulated_sequences/NeuralTextures/c23/flow",
        ],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // 2,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
    )

    # model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    conv1 = nn.Conv2d(
        2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    fc = nn.Linear(2048, 2, bias=True)
    model.conv1 = conv1
    model.fc = fc
    model = model.to("cuda")
    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Number of parameters:", n_parameters)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = StepLR(opt, step_size=int(args.num_epoch / 4 * 3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epoch):
        train(model, criterion, opt, train_loader, writer, epoch, args)
        lr_scheduler.step()

        save_model_path = Path(args.dir_path, "checkpoints", "last.pth")
        torch.save(model.state_dict(), save_model_path)


def train(model, criterion, opt, train_loader, writer, epoch: int, args):
    model.train()
    metric_logger = MetricLogger(
        print_freq=args.print_freq, writer=writer, writer_prefix="train/"
    )

    for step, data in metric_logger.log_every(train_loader, header=f"[train {epoch}]"):
        flow = data["flow"].to(args.device).float()
        target = data["label"].to(args.device).long()
        output = model(flow)
        loss = criterion(output, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_value = loss.item()
        acc = compute_accuray(F.log_softmax(output, dim=1), target)

        metric_logger.update(loss=loss_value, acc=acc)

    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())


if __name__ == "__main__":
    args = get_args()
    main(args)
