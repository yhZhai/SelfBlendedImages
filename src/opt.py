import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from rich import print


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="session_name")

    parser.add_argument("--seed", type=int, default=42)

    # data
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/FaceForensics++/original_sequences/youtube/c23/frames/",
    )
    # eval
    parser.add_argument("--eval_freq", type=int, default=5)

    # train
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # save
    parser.add_argument("--save_root_path", type=str, default="work_dir")
    parser.add_argument("--session", type=str, default="twinsbaseline")
    parser.add_argument("--num_weight", type=int, default=5)

    # misc
    parser.add_argument("--print_freq", type=int, default=20)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    # automatically set parameters
    if len(sys.argv) > 1:
        arguments = sys.argv[1:]
        arguments = list(
            map(lambda x: x.replace("--", ""), filter(lambda x: "--" in x, arguments))
        )
        params = []
        for argument in arguments:
            if not argument in [
                "suffix",
                "save_root_path",
                "dataset",
                "source",
                "resume",
                "num_workers",
                "eval_freq",
                "print_freq",
                "lr_steps",
                "rgb_resume",
                "srm_resume",
                "bayar_resume",
                "teacher_resume",
                "occ",
                "load",
                "amp_opt_level",
                "val_shuffle",
                "tile_size",
                "modality",
            ]:
                try:
                    value = (
                        str(eval("args.{}".format(argument.split("=")[0])))
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "-")
                        .replace(",", "")
                    )
                    params.append(
                        argument.split("=")[0].replace("_", "").replace(" ", "")
                        + "="
                        + value
                    )
                except:
                    logger.warning("Unknown argument: {}".format(argument))
        test_name = "_".join(params)

    else:
        test_name = ""

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.timestamp = timestamp
    args.dir_name = timestamp + "_" + args.session
    if test_name != "":
        args.dir_name += "_" + test_name
    args.dir_path = Path(args.save_root_path, args.dir_name).as_posix()
    print(f"Saving to {args.dir_path}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args
