from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet50_Weights, resnet50

from datasets.flow_dataset import FlowDataset, FlowValDataset
from opt import get_args
from train_sbi import compute_accuray
from utils.misc import MetricLogger, setup_env
from sklearn.metrics import confusion_matrix, roc_auc_score


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

    cdf_val_dataset = FlowValDataset("cdf")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
    )

    cdf_val_loader = torch.utils.data.DataLoader(
        cdf_val_dataset,
        batch_size=1,
        shuffle=False,
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
    model = model.to(args.device)
    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Number of parameters:", n_parameters)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = StepLR(opt, step_size=int(args.num_epoch / 4 * 3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epoch):
        train(model, criterion, opt, train_loader, writer, epoch, args)
        lr_scheduler.step()

        # if epoch % args.eval_freq == 0:
        #     val(model, criterion, cdf_val_loader, writer, epoch, args, "cdf")

        save_model_path = Path(args.dir_path, "checkpoints", "last.pth")
        torch.save(model.state_dict(), save_model_path)


def val(model, criterion, val_loader, writer, epoch: int, args, dataset_name: str):
    model.eval()
    metric_logger = MetricLogger(
        print_freq=args.print_freq, writer=writer, writer_prefix=f"[{dataset_name}]val/"
    )

    labels = []
    scores = []
    with torch.no_grad():
        for step, data in metric_logger.log_every(
            val_loader, header=f"[{dataset_name} val {epoch}]"
        ):
            flow = torch.cat(data["flow"], dim=0).to(args.device).float()
            # flow = data["flow"].to(args.device).float()
            target = data["label"].to(args.device).long()
            output = model(flow)
            output = output.mean(dim=0)

            labels.append(target.item())
            scores.append(output[1].item())

            acc = compute_accuray(F.log_softmax(output.unsqueeze(0), dim=1), target)

            metric_logger.update(acc=acc)

    auc = roc_auc_score(labels, scores)
    metric_logger.update(auc=auc)
    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())


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
