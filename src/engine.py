import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils.misc import MetricLogger


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def train(model, criterion, train_loader, writer, epoch: int, args):
    model.train()
    metric_logger = MetricLogger(
        print_freq=args.print_freq, writer=writer, writer_prefix="train/"
    )

    for step, data in metric_logger.log_every(train_loader, header=f"[train {epoch}]"):
        img = data["img"].to(args.device, non_blocking=True).float()
        target = data["label"].to(args.device, non_blocking=True).long()
        output = model.training_step(img, target)
        loss = criterion(output, target)
        loss_value = loss.item()
        acc = compute_accuray(F.log_softmax(output, dim=1), target)

        metric_logger.update(loss=loss_value, acc=acc)

    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())


def eval(model, criterion, val_loader, writer, epoch: int, args):
    model.eval()
    output_list, target_list = [], []
    metric_logger = MetricLogger(
        print_freq=args.print_freq, writer=writer, writer_prefix="val/"
    )

    for step, data in metric_logger.log_every(val_loader, header=f"[val {epoch}]"):
        img = data["img"].to(args.device, non_blocking=True).float()
        target = data["label"].to(args.device, non_blocking=True).long()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)

        loss_value = loss.item()
        acc = compute_accuray(F.log_softmax(output, dim=1), target)

        output_list += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
        target_list += target.cpu().data.numpy().tolist()

        metric_logger.update(loss=loss_value, acc=acc)

    val_auc = roc_auc_score(target_list, output_list)
    metric_logger.update(auc=val_auc)

    metric_logger.write_tensorboard(epoch)
    print("Average status:")
    print(metric_logger.stat_table())
    return val_auc
