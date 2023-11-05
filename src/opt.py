from datetime import datetime
import argparse
from pathlib import Path

from rich import print
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="session_name")

    parser.add_argument("--seed", type=int, default=42)

    # data
    parser.add_argument("--image_size", type=int, default=380)
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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # save
    parser.add_argument("--save_root_path", type=str, default="work_dir")
    parser.add_argument("--session", type=str, default="flow")
    parser.add_argument("--num_weight", type=int, default=5)

    # misc
    parser.add_argument("--print_freq", type=int, default=20)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.timestamp = timestamp
    args.dir_name = timestamp + "_" + args.session
    args.dir_path = Path(args.save_root_path, args.dir_name).as_posix()
    print(f"Saving to {args.dir_path}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args
