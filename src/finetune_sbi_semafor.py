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
from engine import eval, train
from utils.misc import MetricLogger
from datasets.frame_dataset import FrameDataset


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    writer = setup_env(args)

    train_dataset = FrameDataset(
        real_path_list=[
            "data/eval4/PRISTINE",
        ],
        fake_path_list=[
            "data/eval4/DAGAN",
            "data/eval4/FOMM",
            "data/eval4/LIA",
            "data/eval4/MAXINE",
            "data/eval4/STYLEHEAT",
            "data/eval4/TPS",
        ],
        image_size=args.image_size,
        multiclass=False,
    )
    val_dataset = SBI_Dataset(
        phase="val", image_size=args.image_size, dataset_path=args.dataset_path
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=False,
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

    state_dict = "FFc23.tar"
    state_dict = torch.load(state_dict, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False)

    lr_scheduler = LinearDecayLR(
        model.optimizer, args.num_epoch, int(args.num_epoch / 4 * 3)
    )

    criterion = nn.CrossEntropyLoss()

    last_val_auc = 0
    weight_dict = {}
    for epoch in range(args.num_epoch):
        np.random.seed(args.seed + epoch)
        train(model, criterion, train_loader, writer, epoch, args)
        lr_scheduler.step()

        if not (epoch % args.eval_freq == 0 or epoch == args.num_epoch - 1):
            continue

        np.random.seed(args.seed)
        val_auc = eval(model, criterion, val_loader, writer, epoch, args)

        latest_save_model_path = Path(
            args.dir_path, "checkpoints", "latest.tar",
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "epoch": epoch,
            },
            latest_save_model_path,
        )

        save_model_path = Path(
            args.dir_path, "checkpoints", "{}_{:.4f}_val.tar".format(epoch + 1, val_auc)
        )
        if len(weight_dict) < args.num_weight:
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


if __name__ == "__main__":
    args = get_args()
    main(args)
