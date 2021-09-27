#python main.py --data-name DrugBank --max-nodes-per-hop 200
#python main.py --testing --no-train --data-name DGIdb --max-nodes-per-hop 200
#python main.py --testing --no-train --probe --data-name IDrugBank --max-nodes-per-hop 200
from operator import mod
import torch
import numpy as np

import os
import os.path
import random
import argparse
import warnings

import scipy.io as sio
import scipy.sparse as ssp

import sys, copy, math, time, pdb, warnings, traceback

from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train import *
from models import *
from parser import get_basic_configs
from utils import load_data_from_database

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
# used to traceback which code cause warnings, can delete
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def logger(info, model, optimizer):
    epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
            epoch, train_loss, test_rmse))
    if type(epoch) == int and epoch % args.save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(
            args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        )
        if model is not None:
            torch.save(model.state_dict(), model_name)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_name)

args = get_basic_configs()

args.nums = 25

rating_map, post_rating_map = None, None
if args.standard_rating:
    rating_map = {x: (x-1)//20+1 for x in range(1, 101)}


args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
args.res_dir = os.path.join(
    args.file_dir, 'outputs/{}_{}_{}'.format(
        args.data_name, args.save_appendix, val_test_appendix
    )
)
args.model_pos = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(args.epochs))

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 


# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
os.system(f'cp *.py {args.res_dir}/')
os.system(f'cp *.sh {args.res_dir}/')
print('Python files: *.py and *.sh is saved.')

if args.use_features:
    datasplit_path = 'raw_data/' + args.data_name + '/withfeatures.pickle'
else:
    datasplit_path = 'raw_data/' + args.data_name + '/nofeatures.pickle'

(
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
    val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
    test_v_indices, class_values, u_dict, v_dict
) = load_data_from_database(args.data_name, mode=args.mode, testing=args.testing, rating_map=rating_map)

print('All ratings are:')
print(class_values)
'''
Explanations of the above preprocessing:
    class_values are all the original continuous ratings, e.g. 0.5, 2...
    They are transformed to rating labels 0, 1, 2... acsendingly.
    Thus, to get the original rating from a rating label, apply: class_values[label]
    Note that train_labels etc. are all rating labels.
    But the numbers in adj_train are rating labels + 1, why? Because to accomodate 
    neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
    If testing=True, adj_train will include both train and val ratings, and all train 
    data will be the combination of train and val.
'''


number_of_u, number_of_v = len(u_dict), len(v_dict)
if args.use_features:
    u_features, v_features = u_features.toarray(), v_features.toarray()
    n_features = u_features.shape[1] + v_features.shape[1]
    print('Number of user features {}, item features {}, total features {}'.format(
        u_features.shape[1], v_features.shape[1], n_features))
else:
    u_features, v_features = None, None
    n_features = 0


if args.debug:  # use a small number of data to debug
    num_data = 1000
    train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
    val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
    test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]


train_indices = (train_u_indices, train_v_indices)
val_indices = (val_u_indices, val_v_indices)
test_indices = (test_u_indices, test_v_indices)
print('#train: %d, #val: %d, #test: %d' % (
    len(train_u_indices), 
    len(val_u_indices), 
    len(test_u_indices), 
))

ALL = False
if args.probe:
    u_index, v_index = np.hstack([train_u_indices, test_u_indices]), np.hstack([train_v_indices, test_v_indices])
    exits_tuple = set(zip(u_index, v_index))

    # if args.mode == 'inductive':
    #     test_u_index = np.unique(test_u_indices)
    #     probe_u_idx = np.random.randint(0, len(test_u_index), 10000)
    # else:
    #     probe_u_idx = np.random.randint(0, number_of_u, 10000)
    # probe_v_idx = np.random.randint(0, number_of_v, 10000)
    if ALL:
        u_i = np.arange(number_of_u)
        v_i = np.arange(number_of_v)
        probe_u_idx = np.repeat(u_i, number_of_v)
        probe_v_idx = np.tile(v_i, number_of_u)
    else:
        u_i = np.unique(test_u_indices)
        v_i = np.arange(number_of_v)
        probe_u_idx = np.repeat(u_i, number_of_v)
        probe_v_idx = np.tile(v_i, len(u_i))
    probe_tuple = set(zip(probe_u_idx, probe_v_idx))

    probe_u_indices, probe_v_indices = [], []
    probe_list = list(probe_tuple - exits_tuple)
    for probe_pair in probe_list:
        probe_u_indices.append(probe_pair[0])
        probe_v_indices.append(probe_pair[1])

    probe_u_indices, probe_v_indices = np.array(probe_u_indices), np.array(probe_v_indices)
    probe_indices = (probe_u_indices, probe_v_indices)
    probe_labels = np.ones(len(probe_u_indices), dtype=int)

    res_path = os.path.join(args.res_dir, "predictions_{}_{}_full.csv".format(args.data_name, args.nums))
    drug_id = [u_dict[i] for i in probe_u_indices]
    gene_id = [v_dict[i] for i in probe_v_indices]
    res_df = pd.DataFrame({'Drug': drug_id, 'Gene': gene_id})
    res_df.to_csv(res_path, index=False)

    print('#probe: %d' % (
        len(probe_u_indices) 
    ))

'''
    Extract enclosing subgraphs to build the train/test or train/val/test graph datasets.
    (Note that we must extract enclosing subgraphs for testmode and valmode separately, 
    since the adj_train is different.)
'''
train_graphs, val_graphs, test_graphs = None, None, None
data_combo = (args.data_name, args.data_appendix, val_test_appendix)

if args.reprocess:
    # if reprocess=True, delete the previously cached data and reprocess.
    if os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
        rmtree('data/{}{}/{}/train'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/val'.format(*data_combo)):
        rmtree('data/{}{}/{}/val'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/test'.format(*data_combo)):
        rmtree('data/{}{}/{}/test'.format(*data_combo))


# create dataset, either dynamically extract enclosing subgraphs, 
# or extract in preprocessing and save to disk.
dataset_class = 'MyDynamicDataset' if args.dynamic_train else 'MyDataset'
train_graphs = eval(dataset_class)(
    'data/{}{}/{}/train'.format(*data_combo),
    adj_train, 
    train_indices, 
    train_labels, 
    args.hop, 
    args.sample_ratio, 
    args.max_nodes_per_hop, 
    u_features, 
    v_features, 
    class_values, 
    max_num=args.max_train_num
)
dataset_class = 'MyDynamicDataset' if args.dynamic_test else 'MyDataset'
test_graphs = eval(dataset_class)(
    'data/{}{}/{}/test'.format(*data_combo),
    adj_train, 
    test_indices, 
    test_labels, 
    args.hop, 
    args.sample_ratio, 
    args.max_nodes_per_hop, 
    u_features, 
    v_features, 
    class_values, 
    max_num=args.max_test_num
)
if not args.testing:
    dataset_class = 'MyDynamicDataset' if args.dynamic_val else 'MyDataset'
    val_graphs = eval(dataset_class)(
        'data/{}{}/{}/val'.format(*data_combo),
        adj_train, 
        val_indices, 
        val_labels, 
        args.hop, 
        args.sample_ratio, 
        args.max_nodes_per_hop, 
        u_features, 
        v_features, 
        class_values, 
        max_num=args.max_val_num
    )

# Determine testing data (on which data to evaluate the trained model
if not args.testing: 
    test_graphs = val_graphs

print('Used #train graphs: %d, #test graphs: %d' % (
    len(train_graphs), 
    len(test_graphs), 
))

if args.probe:
    dataset_class = 'MyDataset'
    # os.remove('data/{}{}/{}/probe/processed/data.pt'.format(*data_combo))
    probe_graph = eval(dataset_class)(
        'data/{}{}/{}/probe'.format(*data_combo),
        adj_train, 
        probe_indices, 
        probe_labels, 
        args.hop, 
        args.sample_ratio, 
        args.max_nodes_per_hop, 
        u_features, 
        v_features, 
        class_values, 
        max_num=args.max_val_num
    )

'''
    Train and apply the GNN model
'''
num_relations = len(class_values)
multiply_by = 1

model = CoSMIG(
    train_graphs, 
    latent_dim=[args.hidden]*4, 
    num_relations=num_relations, 
    num_bases=4, 
    regression=True, 
    adj_dropout=args.adj_dropout, 
    force_undirected=args.force_undirected, 
    side_features=args.use_features, 
    n_side_features=n_features, 
    multiply_by=multiply_by
)

# args.ARR = 0.0
# model = CoSMIG(
#     train_graphs,
#     hidden_channels=args.hidden,
#     bias=False,
#     regression=True,
#     dropout=args.adj_dropout,
#     side_features=args.use_features,
#     n_side_features=n_features,
#     multiply_by=multiply_by
# )

if not args.no_train:
    train_multiple_epochs(
        train_graphs,
        test_graphs,
        model,
        args.epochs, 
        args.batch_size, 
        args.lr, 
        lr_decay_factor=args.lr_decay_factor, 
        lr_decay_step_size=args.lr_decay_step_size, 
        weight_decay=0, 
        ARR=args.ARR, 
        test_freq=args.test_freq, 
        logger=logger, 
        continue_from=args.continue_from, 
        res_dir=args.res_dir
    )



if args.testing:
    model.load_state_dict(torch.load(args.model_pos))
    test_once(
        test_graphs,
        model=model,
        batch_size=args.batch_size,
        logger=logger
    )

if not args.testing:
    model.load_state_dict(torch.load(args.model_pos))
    test_once(
        test_graphs,
        model=model,
        batch_size=args.batch_size,
        logger=logger
    )
    test_once(
        val_graphs,
        model=model,
        batch_size=args.batch_size,
        logger=logger
    )


if args.save_results:
    if args.ensemble:
        start_epoch, end_epoch, interval = args.epochs-30, args.epochs, 10

        checkpoints = [
            os.path.join(args.res_dir, 'model_checkpoint%d.pth' %x) 
            for x in range(start_epoch, end_epoch+1, interval)
        ]
        epoch_info = 'ensemble of range({}, {}, {})'.format(
            start_epoch, end_epoch, interval
        )

        for idx, checkpoint in enumerate(checkpoints):
            model.load_state_dict(torch.load(checkpoint))
            save_test_results(
                model=model,
                graphs=test_graphs,
                res_dir=args.res_dir,
                data_name=args.data_name+'_epoch'+str(idx*interval+start_epoch),
                mode='test'
            )
    else:
        model.load_state_dict(torch.load(args.model_pos))
        save_test_results(
            model=model,
            graphs=test_graphs,
            res_dir=args.res_dir,
            data_name=args.data_name
        )

if args.visualize:
    model.load_state_dict(torch.load(args.model_pos))
    visualize(
        model, 
        probe_graph, 
        args.res_dir, 
        args.data_name, 
        class_values, 
        sort_by='prediction'
    )

if args.probe:
    if True:

        start_epoch, end_epoch, interval = args.epochs-30, args.epochs, 10

        checkpoints = [
            os.path.join(args.res_dir, 'model_checkpoint%d.pth' %x) 
            for x in range(start_epoch, end_epoch+1, interval)
        ]
        epoch_info = 'ensemble of range({}, {}, {})'.format(
            start_epoch, end_epoch, interval
        )

        if True and checkpoints:
            predict(
                model=model,
                graphs=probe_graph,
                res_dir=args.res_dir,
                data_name=args.data_name,
                class_values=class_values,
                num=args.nums,
                sort_by='prediction',
                checkpoints=checkpoints,
                ensemble=True
            )
            # for checkpoint in checkpoints:
            #     model.load_state_dict(torch.load(checkpoint))
            #     predict(model=model, 
            #             graphs=probe_graph, 
            #             res_dir=args.res_dir, 
            #             data_name=args.data_name, 
            #             class_values=class_values, 
            #             num=args.nums, 
            #             sort_by='prediction'
            #         )
    else:
        model.load_state_dict(torch.load(args.model_pos))
        predict(model=model, 
                graphs=probe_graph, 
                res_dir=args.res_dir, 
                data_name=args.data_name, 
                class_values=class_values, 
                num=args.nums, 
                sort_by='prediction'
            )

if True:
    start_epoch, end_epoch, interval = args.epochs-30, args.epochs, 10

    checkpoints = [
        os.path.join(args.res_dir, 'model_checkpoint%d.pth' %x) 
        for x in range(start_epoch, end_epoch+1, interval)
    ]
    epoch_info = 'ensemble of range({}, {}, {})'.format(
        start_epoch, end_epoch, interval
    )

    rmse = test_once(
        test_graphs, 
        model, 
        args.batch_size, 
        logger=None, 
        ensemble=True, 
        checkpoints=checkpoints
    )
    print('Ensemble test rmse is: {:.6f}'.format(rmse))


    eval_info = {
        'epoch': epoch_info,
        'train_loss': 0,
        'test_rmse': rmse,
    }
    logger(eval_info, None, None)