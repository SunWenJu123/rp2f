import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils.args import get_args
from utils.training import train_il
from utils.conf import set_random_seed
import torch




def main():
    args = get_args()
    args.model = 'rp2f'
    args.seed = None
    args.validation = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        set_random_seed(args.seed)


    # seq-tinyimg
    args.dataset = 'seq-tinyimg'
    args.print_freq = 10
    args.n_epochs = 100
    args.classifier = 'linear'
    args.scheduler_step = 90
    args.ssl_leaner = 'moco'

    args.lr = 0.3
    args.clslr = 0.07
    args.batch_size = 1024
    args.weight_decay = 0
    args.momentum = 0
    args.eta = 0        
    args.eps_perturb = 1e-5
    args.lambd = 1e-5      
    args.eps = 1e-8

    args.dataset = 'seq-cifar100'
    args.print_freq = 10
    args.n_epochs = 100
    args.classifier = 'linear'
    args.scheduler_step = 99
    args.ssl_leaner = 'moco'

    args.lr = 0.05
    args.clslr = 0.05
    args.batch_size = 32
    args.weight_decay = 0
    args.eta = 1e-6
    args.lambd = 1e-5
    args.eps = 1e-8
    args.eps_perturb = 0.001

    for conf in [0]:
        print("")
        print("=================================================================")
        print("==========================", "index", ":", conf, "==========================")
        print("=================================================================")
        print("")
        train_il(args)


if __name__ == '__main__':
    main()
