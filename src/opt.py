import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser('NIPS2023 Deepfake detection model', add_help=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--wholetest', action='store_true', default=False)

    parser.add_argument('--load', default='configs/final.yaml',
                        help='Load configuration YAML file.')
    parser.add_argument('--num_class', type=int, default=2)

    # training parameters
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'])
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_freq', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--grad_clip', type=float, default=0.)

    # lr
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')

    # save
    parser.add_argument('--save_root_path', type=str, default='tmp')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--print_freq', type=int, default=100)

    # misc
    parser.add_argument('--seed', type=int, default=42)

    return parser
