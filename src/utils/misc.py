"""
Author: Yuanhao Zhai
Date: 2022-04-04 22:14:59
Copyright (c) 2022 by Yuanhao Zhai <yuanhaozhai@gmail.com>, All Rights Reserved. 
"""
import sys
import os
import subprocess
import time
import datetime
import random
import copy
import json
from typing import Dict
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import prettytable as pt
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from loguru import logger


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.avg: .5f}"


def get_sha():
    """Get git current status"""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    message = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        sha = sha[:8]
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        message = _run(["git", "log", "--pretty=format:'%s'", sha, "-1"]).replace(
            "'", ""
        )
    except Exception:
        pass

    return {"sha": sha, "status": diff, "branch": branch, "prev_commit": message}


def setup_env(opt):
    # deterministic
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # mkdir subdirectories
    dir_path = Path(opt.dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    # # save log
    # sys.stdout = Logger(dir_path / "log.log", sys.stdout)
    # sys.stderr = Logger(dir_path / "error.log", sys.stderr)
    logger.add((dir_path / "log.log").as_posix())

    # save parameters
    params = copy.deepcopy(opt)
    params = vars(params)
    params.pop("device")
    with open((dir_path / Path("params.json")).as_posix(), "w") as f:
        json.dump(params, f)

    # print info
    logger.info(
        "Running on {}, PyTorch version {}, files will be saved at {}".format(
            opt.device, torch.__version__, dir_path.as_posix()
        )
    )
    # for i in range(torch.cuda.device_count()):
    #     logger.info("\t{}:".format(i), torch.cuda.get_device_name(i))
    logger.info(f"Git: {get_sha()}.")

    # return tensorboard summarywriter
    return SummaryWriter("{}/".format(dir_path.as_posix()))


class MetricLogger(object):
    def __init__(
        self, print_freq: int = 10, delimiter=" ", writer=None, writer_prefix: str = ""
    ):
        self.meters = defaultdict(AverageMeter)
        self.print_freq = print_freq
        self.delimiter = delimiter
        self.writer = writer
        self.writer_prefix = writer_prefix

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Unsupport type {type(v)}."
            self.meters[k].update(v)

    def log_every(self, iterable, header=""):
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "iter time: {time}s",
            ]
        )
        for obj in iterable:
            yield i, obj
            iter_time.update(time.time() - end)
            if (i + 1) % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    log_msg.format(
                        i + 1,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                    ).replace("  ", " ")
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.4f}s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

    def write_tensorboard(self, step):
        if self.writer is not None:
            for k, v in self.meters.items():
                self.writer.add_scalar(self.writer_prefix + k, v.avg, step)

    def stat_table(self):
        tb = pt.PrettyTable(field_names=["Metrics", "Values"])
        for name, meter in self.meters.items():
            tb.add_row([name, str(meter)])
        return tb.get_string()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str).replace("  ", " ")


def save_model(path, model: nn.Module, epoch, opt, performance=None):
    if not opt["debug"]:
        try:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "opt": opt,
                    "performance": performance,
                },
                path,
            )
        except Exception as e:
            cprint("Failed to save {} because {}".format(path, str(e)))


def resume_from(model: nn.Module, resume_path: str):
    checkpoint = torch.load(resume_path, map_location="cpu")
    try:
        state_dict = checkpoint["model"]
    except:
        model.load_state_dict(checkpoint)
        return

    performance = checkpoint["performance"]
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        model.load_state_dict(state_dict, strict=False)
        cprint("Failed to load full model because {}".format(str(e)), "red")
        time.sleep(3)
    print(f"{resume_path} model loaded. It performance is")
    if performance is not None:
        for k, v in performance.items():
            print(f"{k}: {v}")
    return checkpoint["opt"]


def update_record(result: Dict, epoch: int, opt, file_name: str = "latest_record"):
    if not opt["debug"]:
        # save txt file
        tb = pt.PrettyTable(field_names=["Metrics", "Values"])
        with open(Path(opt["dir_path"], f"{file_name}.txt").as_posix(), "w") as f:
            f.write(f"Performance at {epoch}-th epoch:\n\n")
            for k, v in result.items():
                tb.add_row([k, "{:.7f}".format(v)])
            f.write(tb.get_string())

        # save json file
        result["epoch"] = epoch
        with open(Path(opt["dir_path"], f"{file_name}.json"), "w") as f:
            json.dump(result, f)
