import torch
import torch.nn as nn

import utils.misc as misc


def train(model: nn.Module, dataloader, optimizer, criterion, epoch: int, writer, opt):
    logger = misc.MetricLogger(writer=writer)
    model.train()
    
    for i, data in logger.log_every(dataloader, print_freq=opt.print_freq):
        optimizer.zero_grad()

        img = data["frames"].to(opt.device).float()[:, 0]
        target = data["label"].to(opt.device).long()
    pass

def evaluate():
    pass
