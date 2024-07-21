from argparse import ArgumentParser

from utils.conf import set_random_seed


def get_args():
    parser = ArgumentParser(description='deep inc params', allow_abbrev=False)
    # 数据集参数
    parser.add_argument('--root', type=str, default='./data/',
                        help='dictionary of dataset')
    parser.add_argument('--transform', type=str, default='default',
                        help='default or pytorch.')
    parser.add_argument('--featureNet', type=str, default=None,
                        help='feature extractor')
    parser.add_argument('--nt', type=int, default=None,
                        help='task number')
    parser.add_argument('--t_c_arr', type=str, default=None,
                        help='class array for each task')
    parser.add_argument('--validation', type=bool, default=False,
                        help='is test with the validation set')
    parser.add_argument('--class_shuffle', type=bool, default=False,
                        help='is random shuffle the classes order')
    parser.add_argument('--task_shuffle', type=bool, default=False,
                        help='is random shuffle the task order')
    parser.add_argument('--keep_ratio', type=float, default=1,
                        help='ratio samples too keep')
    # 模型参数
    parser.add_argument('--backbone', type=str, default='None',
                        help='the backbone of model')
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='number of epoch')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size')
    # 其他参数
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print loss frequency')
    parser.add_argument('--img_dir', type=str, default='img/',
                        help='image dir')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed if None')
    parser.add_argument('--repeat', type=int, default=1,
                        help='repeat number')
    args = parser.parse_known_args()[0]

    if args.seed is not None:
        set_random_seed(args.seed)

    return args