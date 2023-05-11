import os
import random
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

from opt import get_args
from model import Detector
from utils.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR
from utils.misc import setup_env


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    writer = setup_env(args)

    train_dataset = SBI_Dataset(phase="train", image_size=args.image_size, dataset_path=args.dataset_path)
    val_dataset = SBI_Dataset(phase="val", image_size=args.image_size, dataset_path=args.dataset_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // 2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=args.num_worker,
        pin_memory=True,
        worker_init_fn=val_dataset.worker_init_fn,
    )

    model = Detector(args)

    model = model.to("cuda")

    iter_loss = []
    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []
    lr_scheduler = LinearDecayLR(
        model.optimizer, args.num_epoch, int(args.num_epoch / 4 * 3)
    )

    # logger = log(path=args.dir_path, file="losses.logs")

    criterion = nn.CrossEntropyLoss()

    last_val_auc = 0
    weight_dict = {}
    n_weight = 5
    for epoch in range(args.num_epoch):
        # np.random.seed(seed + epoch)
        train_loss = 0.0
        train_acc = 0.0
        model.train(mode=True)
        for step, data in enumerate(tqdm(train_loader)):
            img = data["img"].to(args.device, non_blocking=True).float()
            target = data["label"].to(args.device, non_blocking=True).long()
            output = model.training_step(img, target)
            loss = criterion(output, target)
            loss_value = loss.item()
            iter_loss.append(loss_value)
            train_loss += loss_value
            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            train_acc += acc
        lr_scheduler.step()
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))

        log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
            epoch + 1,
            args.num_epoch,
            train_loss / len(train_loader),
            train_acc / len(train_loader),
        )

        model.train(mode=False)
        val_loss = 0.0
        val_acc = 0.0
        output_dict = []
        target_dict = []
        # np.random.seed(seed)
        for step, data in enumerate(tqdm(val_loader)):
            img = data["img"].to(args.device, non_blocking=True).float()
            target = data["label"].to(args.device, non_blocking=True).long()

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, target)

            loss_value = loss.item()
            iter_loss.append(loss_value)
            val_loss += loss_value
            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            val_acc += acc
            output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
            target_dict += target.cpu().data.numpy().tolist()
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))
        val_auc = roc_auc_score(target_dict, output_dict)
        log_text += "val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
            val_loss / len(val_loader), val_acc / len(val_loader), val_auc
        )

        save_model_path = Path(
            args.dir_path, "checkpoints", "{}_{:.4f}_val.tar".format(epoch + 1, val_auc)
        )
        if len(weight_dict) < n_weight:
            weight_dict[save_model_path] = val_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_model_path,
            )
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        elif val_auc >= last_val_auc:
            for k in weight_dict:
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_auc
                    break
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_model_path,
            )
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        # logger.info(log_text)


if __name__ == "__main__":
    args = get_args()
    main(args)
