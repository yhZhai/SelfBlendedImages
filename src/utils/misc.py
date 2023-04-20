import sys
import os
import subprocess
import time
import datetime
from collections import defaultdict
from shutil import copy2
import random
import copy
import json
import warnings
import math
from typing import Dict
import signal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import prettytable as pt
from termcolor import cprint
import wandb


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


def set_determinsitic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_env(opt, file_list=None):
    if opt.eval or opt.debug:
        opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)
        return None

    # wandb
    wholetest = "wholetest" in opt.dir_name.lower()
    if wholetest:
        cprint("wholetest mode", "red")
    wandb_config = copy.deepcopy(vars(opt))
    [
        wandb_config.pop(x, None)
        for x in ["eval", "debug", "time_stamp", "dir_name", "save_root_path", "suffix"]
    ]
    wandb.init(
        project="ICLR2024-Deepfake",
        entity="had0w",
        name=opt.dir_name,
        tags=["wholetest"] if wholetest else None,
        config=wandb_config,
    )
    exclude_list = [
        "tmp",
        "wandb",
        "cache",
        "obsolete",
        "lib",
        "pretrained",
        "__pycache__",
        "data",
    ]
    wandb.run.log_code(
        "..", exclude_fn=lambda path: any(sub_path in path for sub_path in exclude_list)
    )

    dir_name = opt.dir_name
    save_root_path = opt.save_root_path
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    set_determinsitic(opt.seed)

    # mkdir subdirectories
    run_env = "run_env"
    checkpoint = "checkpoint"
    if not os.path.exists(os.path.join(save_root_path, dir_name)):
        os.mkdir(os.path.join(save_root_path, dir_name))
        os.mkdir(os.path.join(save_root_path, dir_name, run_env))
        os.mkdir(os.path.join(save_root_path, dir_name, checkpoint))

    # save log
    sys.stdout = Logger(os.path.join(save_root_path, dir_name, "log.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(save_root_path, dir_name, "error.log"), sys.stderr)

    # save file lists
    if file_list is not None:
        file_list = [file_list] if type(file_list) is not list else file_list
        for item in file_list:
            if os.path.isdir(item):  # TODO subdirectory
                if not os.path.exists(
                    os.path.join(save_root_path, dir_name, run_env, item)
                ):
                    os.mkdir(os.path.join(save_root_path, dir_name, run_env, item))
                for file in filter(lambda x: x.endswith(".py"), os.listdir(item)):
                    copy2(
                        os.path.join(item, file),
                        os.path.join(save_root_path, dir_name, "run_env", item, file),
                    )
            else:
                file_name = item.split("/")[-1]
                copy2(item, os.path.join(save_root_path, dir_name, run_env, file_name))

    # save parameters
    params = copy.deepcopy(vars(opt))
    params.pop("device")
    with open(os.path.join(save_root_path, dir_name, "params.json"), "w") as f:
        json.dump(params, f)

    # print info
    print(
        "Running on {}, PyTorch version {}, files will be saved at {}".format(
            opt.device, torch.__version__, os.path.join(save_root_path, dir_name)
        )
    )
    print("Devices:")
    for i in range(torch.cuda.device_count()):
        print("    {}:".format(i), torch.cuda.get_device_name(i))
    print(f"Git: {get_sha()}.")

    # return tensorboard summarywriter
    return SummaryWriter("{}/{}/".format(opt.save_root_path, opt.dir_name))


class MetricLogger(object):
    def __init__(self, delimiter=" ", writer=None, suffix=None):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter
        self.writer = writer
        self.suffix = suffix

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Unsupport type {type(v)}."
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def get_meters(self):
        result = {}
        for k, v in self.meters.items():
            result[k] = v.avg
        return result

    def prepend_subprefix(self, subprefix: str):
        old_keys = list(self.meters.keys())
        for k in old_keys:
            self.meters[k.replace("/", f"/{subprefix}")] = self.meters[k]
        for k in old_keys:
            del self.meters[k]

    def log_every(self, iterable, print_freq=10, header=""):
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
            if (i + 1) % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
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
        print(
            "{} Total time: {} ({:.4f}s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

    def write_tensorboard(self, step):
        if self.writer is not None:
            for k, v in self.meters.items():
                # if self.suffix:
                #     self.writer.add_scalar(
                #         '{}/{}'.format(k, self.suffix), v.avg, step)
                # else:
                self.writer.add_scalar(k, v.avg, step)

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
    if not opt.debug:
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
    state_dict = checkpoint["model"]
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


def update_record(result: Dict, epoch: int, opt, file_name: str = "latest_record"):
    if not opt.debug:
        # save txt file
        tb = pt.PrettyTable(field_names=["Metrics", "Values"])
        with open(
            os.path.join(opt.save_root_path, opt.dir_name, f"{file_name}.txt"), "w"
        ) as f:
            f.write(f"Performance at {epoch}-th epoch:\n\n")
            for k, v in result.items():
                tb.add_row([k, "{:.7f}".format(v)])
            f.write(tb.get_string())

        # save json file
        result["epoch"] = epoch
        with open(
            os.path.join(opt.save_root_path, opt.dir_name, f"{file_name}.json"), "w"
        ) as f:
            json.dump(result, f)
