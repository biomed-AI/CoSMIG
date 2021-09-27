
import torch
import random
import argparse
import numpy as np

def get_basic_configs():

    # Arguments
    parser = argparse.ArgumentParser(description='CoSMIG')
    # general settings
    parser.add_argument('--testing', action='store_true', default=False,
                        help='if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='if set, skip the training and directly perform the \
                        transfer/ensemble/visualization')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='turn on debugging mode which uses a small number of data')
    parser.add_argument('--data-name', default='ml_100k', help='dataset name')
    parser.add_argument('--mode', default='transductive', 
                        help='what to append to save-names when saving datasets')
    parser.add_argument('--data-appendix', default='', 
                        help='what to append to save-names when saving datasets')
    parser.add_argument('--save-appendix', default='', 
                        help='what to append to save-names when saving results')
    parser.add_argument('--save-results', action='store_true', default=False,
                        help='')
    parser.add_argument('--max-train-num', type=int, default=None, 
                        help='set maximum number of train data to use')
    parser.add_argument('--max-val-num', type=int, default=None, 
                        help='set maximum number of val data to use')
    parser.add_argument('--max-test-num', type=int, default=None, 
                        help='set maximum number of test data to use')
    parser.add_argument('--seed', type=int, default=8888, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                        help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                        valid only for ml_1m and ml_10m')
    parser.add_argument('--reprocess', action='store_true', default=False,
                        help='if True, reprocess data instead of using prestored .pkl data')
    parser.add_argument('--dynamic-train', action='store_true', default=False,
                        help='extract training enclosing subgraphs on the fly instead of \
                        storing in disk; works for large datasets that cannot fit into memory')
    parser.add_argument('--dynamic-test', action='store_true', default=False)
    parser.add_argument('--dynamic-val', action='store_true', default=False)
    parser.add_argument('--keep-old', action='store_true', default=False,
                        help='if True, do not overwrite old .py files in the result folder')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save model states every # epochs ')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden size')

    # subgraph extraction settings
    parser.add_argument('--hop', default=3, metavar='S', 
                        help='enclosing subgraph hop number')
    parser.add_argument('--sample-ratio', type=float, default=1.0, 
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max-nodes-per-hop', default=10000, 
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    parser.add_argument('--use-features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    # edge dropout settings
    parser.add_argument('--adj-dropout', type=float, default=0.1, 
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False, 
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    # optimization settings
    parser.add_argument('--continue-from', type=int, default=None, 
                        help="from which epoch's checkpoint to continue training")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay-step-size', type=int, default=50,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='batch size during training')
    parser.add_argument('--test-freq', type=int, default=1, metavar='N',
                        help='test every n epochs')
    parser.add_argument('--ARR', type=float, default=0.001, 
                        help='The adjacenct rating regularizer. If not 0, regularize the \
                        differences between graph convolution parameters W associated with\
                        adjacent ratings')
    # transfer learning, ensemble, and visualization settings
    parser.add_argument('--transfer', default='',
                        help='if not empty, load the pretrained models in this path')
    parser.add_argument('--num-relations', type=int, default=5,
                        help='if transfer, specify num_relations in the transferred model')
    parser.add_argument('--multiply-by', type=int, default=1,
                        help='if transfer, specify how many times to multiply the predictions by')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='if True, load a pretrained model and do visualization exps')
    parser.add_argument('--probe', action='store_true', default=False,
                        help='if True, load a pretrained model and probe dataset.')
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='if True, load a series of model checkpoints and ensemble the results')
    parser.add_argument('--standard-rating', action='store_true', default=False,
                        help='if True, maps all ratings to standard 1, 2, 3, 4, 5 before training')
    # sparsity experiment settings
    parser.add_argument('--ratio', type=float, default=1.0,
                        help="For ml datasets, if ratio < 1, downsample training data to the\
                        target ratio")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)
    
    return args
