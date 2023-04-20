from datetime import datetime
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils.ffpp_video_dataset import FFPPVideoDataset
from utils.scheduler import LinearDecayLR
from utils.logs import log
from utils.funcs import load_json
from utils.misc import set_determinsitic
from model import Detector
from models.st_transformer import SpatioTemporalTransformer
from opt import get_argument_parser
from engine import train, evaluate


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(opt):
    writer = None

    set_determinsitic()

    train_dataset = FFPPVideoDataset(
        phase="train", image_size=opt.image_size, n_frames=opt.num_frame, verbose=opt.verbose)
    val_dataset = FFPPVideoDataset(
        phase="val", image_size=opt.image_size, n_frames=opt.num_frame, verbose=opt.verbose)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0 if opt.debug else opt.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0 if opt.debug else opt.num_workers,
        pin_memory=True,
    )

    model = SpatioTemporalTransformer(num_layers=3).to(opt.device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {}".format(n_param))

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_steps, gamma=0.1)

    criterion = nn.CrossEntropyLoss().to(opt.device)

    if opt.resume:
        pass  # TODO

    if opt.eval:
        pass  # TODO
        return
    
    for epoch in range(opt.epochs):
        train(model, train_loader, optimizer, criterion, epoch, writer, opt)
        scheduler.step()
        

    iter_loss = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    val_losses = []
    last_loss = 99999

    # now = datetime.now()
    # save_path = (
    #     "output/{}_".format(args.session_name)
    #     + now.strftime(os.path.splitext(os.path.basename(args.config))[0])
    #     + "_"
    #     + now.strftime("%m_%d_%H_%M_%S")
    #     + "/"
    # )
    # os.mkdir(save_path)
    # os.mkdir(save_path + "weights/")
    # os.mkdir(save_path + "logs/")
    # logger = log(path=save_path + "logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss()

    last_auc = 0
    last_val_auc = 0
    weight_dict = {}
    n_weight = 5
    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.0
        train_acc = 0.0
        model.train(mode=True)
        for step, data in enumerate(tqdm(train_loader, desc=f"[{epoch}] train")):
            landmarks = data["landmarks"].to(device).float()
            patches = data["patches"].to(device).float()
            out = stt(landmarks, patches)

            img = data["frames"].to(device).float()[:, 0]
            target = data["label"].to(device).long()
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
            n_epoch,
            train_loss / len(train_loader),
            train_acc / len(train_loader),
        )

        model.train(mode=False)
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
                    "optimizer": model.optimizer.state_dict(),
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
                    "optimizer": model.optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_model_path,
            )
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        logger.info(log_text)


if __name__ == "__main__":
    parser = get_argument_parser()
    opt = parser.parse_args()
    opt.device = torch.device("cuda")

    main(opt)
