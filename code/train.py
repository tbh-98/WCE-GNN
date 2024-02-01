import os
import time
import random
import torch
import dgl
import argparse
import copy

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *

from convert_datasets_to_pygDataset import dataset_Hypergraph

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            #print(f'Run {run + 1:02d}:')
            #print(f'Highest Train: {result[:, 0].max():.2f}')
            #print(f'Highest Valid: {result[:, 1].max():.2f}')
            #print(f'  Final Train: {result[argmax, 0]:.2f}')
            #print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            # result = 100 * torch.tensor(self.results)

            best_results = []
            best_es = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                best_e = r[:, 1].argmax().item()
                best_es.append(best_e)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            #print(f'All runs:')
            r = best_result[:, 0]
            #print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            #print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            #print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            #print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3], best_es

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, x, y, split_idx, eval_func, hf, g, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(x, g)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main part of the training ---
# # Part 0: Parse arguments
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True 
    torch.cuda.current_device()
    torch.cuda._initialized = True

def get_g(edge_index0, args, device, num_nodes, num_hyperedges): # use check diagonal to make sure no isolated nodes for other ce methods
    edge_index0[0] = edge_index0[0] - edge_index0[0].min()
    edge_index0[1] = edge_index0[1] - edge_index0[1].min()
    
    Hrow, Hcol = edge_index0[0], edge_index0[1]
    Hval = np.ones_like(Hrow, dtype = np.int8)

    H = sp.coo_matrix((Hval, (Hrow, Hcol)), (num_nodes, int(num_hyperedges)), dtype=np.float64)

    if(args.nh):
        De = np.sum(H, axis=0)
        De = De * 1.0
        De = np.reciprocal(De, where= De!=0)
        De = sp.diags(De.A1)
        H = H @ De @ H.T
    else:
        H = H @ H.T
        H = (H > 0) * 1.0
    H.setdiag(1)
    Dv = np.sum(H, axis=0)
    Dv = Dv * 1.0
    Dv = np.sqrt(Dv, where= Dv!=0)
    Dv = np.reciprocal(Dv, where= Dv!=0)
    Dv = sp.diags(Dv.A1)
    H = Dv @ H @ Dv
    if(args.method == 'GCNII'):
        edge_index = H.nonzero()
        coo_data = torch.FloatTensor(H.data)
        coo_rows = torch.LongTensor(edge_index[0])
        coo_cols = torch.LongTensor(edge_index[1])
        H = torch.sparse.FloatTensor(torch.stack([coo_rows, coo_cols]), coo_data, (int(num_nodes), int(num_nodes)))
        H = H.to(device)
        return H, None
    else:
        return H, None

def generate_dgl_g(g, device):
    edge_index0 = g.nonzero()
    values = g[g != 0]
    values = values.T
    values = torch.tensor(values)
    g = dgl.graph((torch.tensor(edge_index0[0]), torch.tensor(edge_index0[1])))
    g.edata['w'] = values
    g = g.to(device)
    return g
"""

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='GAT')
    parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1,0,1,2,3,4,5,6,7], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--cls_layers', default=1, type=int) 
    parser.add_argument('--hidden', default=512,
                        type=int)
    parser.add_argument('--cls_hidden', default=64,
                        type=int)
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--nres', action='store_false', dest='res')
    parser.add_argument('--spectral_a', action='store_true', dest='spectral_a')
    parser.add_argument('--spa2', action='store_true', dest='spa2')
    parser.add_argument('--nh', action='store_true', dest='nh')
    parser.add_argument('--lalpha', action='store_true', dest='learn_alpha')
    parser.add_argument('--llamda', action='store_true', dest='learn_lamda')
    parser.add_argument('--conv_type', default='GATv2Conv')
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--in_channels_h', default=0, type=int)  # Placeholder
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder 
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    parser.add_argument('--act', default='relu', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--variant', action='store_true', dest='variant')
    #     Args for Attentions: GAT and SetGNN 
    parser.add_argument('--k', default=1, type=int)  # Placeholder
    parser.add_argument('--heads', default=2, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--alpha', default=15.0, type=float)
    parser.add_argument('--lamda', default=0.5, type=float)
    parser.add_argument('--step_size', default=1000000000, type=int)
    parser.add_argument('--gamma', default=100000000, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()
    
    
    # # Part 1: Load data
    
    
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed', 'senate-committees-100', 'congress-bills-100']
    
    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100', 'senate-committees-100', 'congress-bills-100']
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100', 'senate-committees-100', 'congress-bills-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
    
    try:
        num_nodes = data.n_x[0]
    except:
        num_nodes=data.n_x
    try:
        num_hyperedges = data.num_hyperedges[0]
    except:
        num_hyperedges = data.num_hyperedges
    
    data = ExtractV2E(data)
    
    #     Get splits
    setup_seed(1000)
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    
    # put things to device
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    edge_index = data.edge_index
    # edge_index[1] = edge_index[1] - data.n_x
    
    
    args.in_channels_h = num_hyperedges
    g, hf = get_g(edge_index, args, device, num_nodes, num_hyperedges)
    if(args.method != 'GCNII'):
        g = generate_dgl_g(g, device)
    # # Part 2: Load model
    
    if(args.method == 'GCNII'):
        model = GCNII(args)
    elif(args.method == 'SGC'):
        model = SGC(args)
    elif(args.method == 'GAT'):
        model = GAT(args)
    elif(args.method == 'GCN'):
        model = GCN(args)
        
    x, y = data.x, data.y
    
    del data
    
    model, x, y = model.to(device), x.to(device), y.to(device)
    args.device = device
    
    num_params = count_parameters(model)
    
    
    # # Part 3: Main. Training + Evaluation
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model.train()
    # print('MODEL:', model)
    
    ### Training loop ###
    runtime_list = []
    for run in tqdm(range(args.runs)):
        setup_seed(run)
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if(args.step_size < 100000000):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
        
        best_val = float(0)
        recorder = 0
        # bar = int((args.epochs * 0.1))
        bar = 200
        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            out = model(x, g)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], y[train_idx])
            loss.backward()
            optimizer.step()
            if(args.step_size < 100000000):
                scheduler.step()
            result = evaluate(model, x, y, split_idx, eval_func, hf, g)
            logger.add_result(run, result[:3])
            recorder = recorder + 1
            if(result[1] > best_val):
                best_val = result[1]
                recorder = 0

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
            if(recorder > bar):
                break

        end_time = time.time()
        runtime_list.append(end_time - start_time)
    
    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val, best_test, best_es = logger.print_statistics()
    
    res_root = 'log'
    if not osp.isdir(res_root):
        os.makedirs(res_root)
    res_root = res_root + f'/{args.method}'
    if not osp.isdir(res_root):
        os.makedirs(res_root)
    
    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}_nh{args.nh}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj: # cls_layers cls_hidden 15 20 25 30 35 40 45 50 55 60 
        cur_line = f'val: {best_val.mean():.3f} ± {best_val.std():.3f}\n'
        cur_line += f'test: {best_test.mean():.3f} ± {best_test.std():.3f}\n'
        cur_line += f'\n'
        print(cur_line)
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    print('All done! Exit python code')
    quit()

