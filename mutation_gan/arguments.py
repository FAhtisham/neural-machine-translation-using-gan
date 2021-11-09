import argparse


def add_general_args(parser):
    parser.add_argument("--seed", default=1, type=int,
                      help="Random seed. (default=1)")
    return parser

def add_generator_model_args(parser):
    parser.add_argument('--encoder-embed-dim', default=512, type=int,
                       help='encoder embedding dimension')
    parser.add_argument('--encoder-layers', default=1, type=int,
                       help='encoder layers [(dim, kernel_size), ...]')
    parser.add_argument('--decoder-embed-dim', default=512, type=int,
                       help='decoder embedding dimension')
    parser.add_argument('--decoder-layers', default=1, type=int,
                       help='decoder layers [(dim, kernel_size), ...]')
    parser.add_argument('--decoder-out-embed-dim', default=512, type=int,
                       help='decoder output dimension')
    parser.add_argument('--encoder-dropout-in', default=0.1, type=float,
                       help='dropout probability for encoder input embedding')
    parser.add_argument('--encoder-dropout-out', default=0.1, type=float,
                       help='dropout probability for encoder output')
    parser.add_argument('--decoder-dropout-in', default=0.1, type=float,
                       help='dropout probability for decoder input embedding')
    parser.add_argument('--decoder-dropout-out', default=0.1, type=float,
                       help='dropout probability for decoder output')
    parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--bidirectional', default=False, type=bool,
                       help='unidirectional or bidirectional encoder')
    return parser
    



def add_discriminator_model_args(parser):
    parser.add_argument('--fixed-max-len', default=50, type=int,
                       help='the max length the discriminator can hold')
    parser.add_argument('--d-sample-size', default=5000, type=int,
                       help='how many data used to pretrain d in one epoch')
    return parser
    
    
    

def add_optimization_args(parser):
    parser.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                        help='force stop training at specified epoch')
    parser.add_argument("--epochs", default=12, type=int,
                        help="Epochs through the data. (default=12)")
    parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--g_optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--d_optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                        help="Learning rate of the optimization. (default=0.1)")
    parser.add_argument("--g_learning_rate", "-glr", default=1e-3, type=float,
                        help="Learning rate of the generator. (default=0.001)")
    parser.add_argument("--d_learning_rate", "-dlr", default=1e-3, type=float,
                        help="Learning rate of the discriminator. (default=0.001)")
    parser.add_argument("--lr_shrink", default=0.5, type=float,
                        help='learning rate shrink factor, lr_new = (lr * lr_shrink)')
    parser.add_argument('--min-g-lr', default=1e-5, type=float, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument('--min-d-lr', default=1e-6, type=float, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Momentum when performing SGD. (default=0.9)")
    parser.add_argument("--use_estop", default=False, type=bool,
                        help="Whether use early stopping criteria. (default=False)")
    parser.add_argument("--estop", default=1e-2, type=float,
                        help="Early stopping criteria on the development set. (default=1e-2)")
    parser.add_argument('--clip-norm', default=5.0, type=float,
                       help='clip threshold of gradients')
    parser.add_argument('--curriculum', default=0, type=int, metavar='N',
                       help='sort batches by source length for first N epochs')
    parser.add_argument('--sample-without-replacement', default=0, type=int, metavar='N',
                       help='If bigger than 0, use that number of mini-batches for each epoch,'
                            ' where each sample is drawn randomly without replacement from the'
                            ' dataset')
    parser.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')

    return parser
