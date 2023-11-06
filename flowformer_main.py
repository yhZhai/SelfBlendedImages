from collections import OrderedDict
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from src.datasets.image_pair_dataset import ImagePairDataset, ValImagePairDataset
from src.FlowFormer import build_flowformer
from src.opt import get_args
from src.utils.misc import MetricLogger, setup_env
from loguru import logger


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(cfg, args):
    writer = setup_env(args)

    # model = build_flowformer(cfg).cuda()
    # state_dict = torch.load("checkpoints/things.pth", map_location="cpu")
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     new_state_dict[k.replace("module.", "")] = v
    # model.load_state_dict(new_state_dict)

    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # model.fc = nn.Linear(2048, 2, bias=True)
    # model = model.cuda()

    model = timm.create_model("twins_svt_large", pretrained=True)
    model.head = nn.Linear(1024, 2, bias=True)
    model = model.cuda()

    dataset = ImagePairDataset(
        "train",
        ["data/FaceForensics++/original_sequences/youtube/c23/frames"],
        [
            "data/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames",
            "data/FaceForensics++/manipulated_sequences/Face2Face/c23/frames",
            "data/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames",
            "data/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames",
        ],
    )

    cdf_val_dataset = ValImagePairDataset("cdf")

    train_loader = torch.utils.data.DataLoader(
        dataset,
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

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(opt, step_size=int(args.num_epoch / 4 * 3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    best_auc_epoch = 0
    for epoch in range(args.num_epoch):
        if epoch % args.eval_freq == 0:
            auc = val(cdf_val_loader, model, epoch, args, writer)
            logger.info(f"Val AUC: {auc} at epoch {epoch}")
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch = epoch
                save_model_path = Path(args.dir_path, "checkpoints", "best.pth")
                torch.save(model.state_dict(), save_model_path)

        train(train_loader, model, opt, criterion, epoch, args, writer)
        lr_scheduler.step()

        save_model_path = Path(args.dir_path, "checkpoints", "last.pth")
        torch.save(model.state_dict(), save_model_path)

        logger.info(f"Best AUC: {best_auc} at epoch {best_auc_epoch}")


def val(val_loader, model, epoch, args, writer=None):
    model.eval()
    metric_logger = MetricLogger(
        print_freq=args.print_freq, writer=writer, writer_prefix="val/"
    )

    labels = []
    scores = []
    with torch.no_grad():
        for step, item in metric_logger.log_every(val_loader, header=f"[val {epoch}]"):
            label = item["label"].cuda()
            image1 = item["first_frame"].cuda()
            # image2 = item["second_frame"].cuda()
            # out = model(image1, image2)
            out = model(image1)
            out = out.mean(dim=0)

            labels.append(label.item())
            scores.append(out[1].item())

            acc = compute_accuray(F.log_softmax(out.unsqueeze(0), dim=1), label)
            metric_logger.update(acc=acc)

    auc = roc_auc_score(labels, scores)
    metric_logger.update(auc=auc)
    metric_logger.write_tensorboard(epoch)
    logger.info("Average status:")
    logger.info(metric_logger.stat_table())
    return auc


def train(train_loader, model, opt, criterion, epoch, args, writer=None):
    model.train()
    metric_logger = MetricLogger(
        print_freq=args.print_freq, writer=writer, writer_prefix="train/"
    )

    for step, item in metric_logger.log_every(train_loader, header=f"[train {epoch}]"):
        label = item["label"].cuda()
        image1 = item["first_frame"].cuda()
        # image2 = item["second_frame"].cuda()
        # pred_flow = model(image1, image2)
        out = model(image1)
        loss = criterion(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = compute_accuray(F.log_softmax(out, dim=1), label)
        metric_logger.update(loss=loss.item(), acc=acc)

    metric_logger.write_tensorboard(epoch)
    logger.info("Average status:")
    logger.info(metric_logger.stat_table())


if __name__ == "__main__":
    from configs.things import get_cfg

    cfg = get_cfg()
    args = get_args()
    main(cfg, args)
