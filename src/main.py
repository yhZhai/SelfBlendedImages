from datetime import datetime
import argparse
import random
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils.ffpp_video_dataset import FFPPVideoDataset
from utils.scheduler import LinearDecayLR
from utils.logs import log
from utils.funcs import load_json
from model import Detector
from networks.xception import TransferModel
from engine import train, evaluate
from opt import get_argument_parser


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    cfg = load_json(args.config)
    n_epoch = cfg["epoch"]

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    # train_dataset = SBI_Dataset(phase="train", image_size=image_size)
    # val_dataset = SBI_Dataset(phase="val", image_size=image_size)
    train_dataset = FFPPVideoDataset(
        phase="train", image_size=image_size, n_frames=1)
    val_dataset = FFPPVideoDataset(
        phase="val", image_size=image_size, n_frames=1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = TransferModel('xception', dropout=0.5, return_fea=False)
    model = model.to("cuda")
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {}".format(n_param))

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(n_epoch / 4 * 3), gamma=0.1)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # for epoch in range(opt.epochs):
    #     train(model, train_loader, optimizer, criterion, epoch, writer, opt)


    iter_loss = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    val_losses = []
    last_loss = 99999

    now = datetime.now()
    save_path = (
        "output/{}_".format(args.session_name)
        + now.strftime(os.path.splitext(os.path.basename(args.config))[0])
        + "_"
        + now.strftime("%m_%d_%H_%M_%S")
        + "/"
    )
    os.mkdir(save_path)
    os.mkdir(save_path + "weights/")
    os.mkdir(save_path + "logs/")
    logger = log(path=save_path + "logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss()

    last_auc = 0
    last_val_auc = 0
    weight_dict = {}
    n_weight = 5
    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for step, data in enumerate(tqdm(train_loader, desc=f"[{epoch}] train")):
            img = data["frames"].to(device).float()[:, 0]
            target = data["label"].to(device).long()
            
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            iter_loss.append(loss_value)
            train_loss += loss_value
            acc = compute_accuray(F.log_softmax(output, dim=1), target)
            train_acc += acc
        scheduler.step()
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))

        log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
            epoch + 1,
            n_epoch,
            train_loss / len(train_loader),
            train_acc / len(train_loader),
        )

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        output_dict = []
        target_dict = []
        np.random.seed(seed)
        for step, data in enumerate(tqdm(val_loader, desc=f"[{epoch}] eval")):
            img = data["frames"].to(device).float()[:, 0]
            target = data["label"].to(device, non_blocking=True).long()

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

        if len(weight_dict) < n_weight:
            save_model_path = os.path.join(
                save_path +
                "weights/", "{}_{:.4f}_val.tar".format(epoch + 1, val_auc)
            )
            weight_dict[save_model_path] = val_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_model_path,
            )
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        elif val_auc >= last_val_auc:
            save_model_path = os.path.join(
                save_path +
                "weights/", "{}_{:.4f}_val.tar".format(epoch + 1, val_auc)
            )
            for k in weight_dict:
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_auc
                    break
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_model_path,
            )
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        logger.info(log_text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="config")
    parser.add_argument("-n", dest="session_name")
    args = parser.parse_args()
    main(args)
