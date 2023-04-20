import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser('Deepfake detection model', add_help=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--wholetest', action='store_true', default=False)

    # parser.add_argument('--load', default='configs/final.yaml',
    #                     help='Load configuration YAML file.')
    parser.add_argument('--num_class', type=int, default=2)

    # model parameters
    parser.add_argument("--patch_size", type=int, default=32)

    # training parameters
    parser.add_argument('--image_size', type=int, default=336, help="only used in dataset")
    parser.add_argument('--num_frame', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'])
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=16)

    # lr
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler_steps', type=int, default=[40, 80], nargs="+",
                        help='epoch interval to decay LR')

    # save
    parser.add_argument('--save_root_path', type=str, default='tmp')
    parser.add_argument('--suffix', type=str, default='')

    # misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', default=False)

    return parser
