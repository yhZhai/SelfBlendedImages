import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.misc as misc


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).detach().cpu().data.numpy()
    tmp = pred_idx == true.detach().cpu().numpy()
    return sum(tmp) / len(pred_idx)


def train(model: nn.Module, dataloader, optimizer, criterion, epoch: int, writer, opt):
    logger = misc.MetricLogger(writer=writer)
    model.train()
    
    for i, data in logger.log_every(dataloader, print_freq=opt.print_freq):
        landmark_loc = data["landmarks"].to(opt.device).float()
        patches = data["patches"].to(opt.device).float()
        label = data["label"].to(opt.device).long()

        optimizer.zero_grad()

        pred = model(landmark_loc, patches)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        acc = compute_accuray(pred, label)
        logger.update(loss=loss, acc=acc)

    logger.write_tensorboard(epoch)
    print('Average status:')
    print(logger.stat_table())


def evaluate():
    pass
