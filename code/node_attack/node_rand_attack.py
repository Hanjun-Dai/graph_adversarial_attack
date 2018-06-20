from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../node_classification' % os.path.dirname(os.path.realpath(__file__)))
from node_utils import load_txt_data, load_binary_data, run_test, load_raw_graph, StaticGraph

from node_attack_common import load_base_model, ModifiedGraph, init_setup

if __name__ == '__main__':    
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    
    features, labels, idx_valid, idx_test, base_model, khop_neighbors = init_setup()
    if cmd_args.meta_test:
        idx_test = idx_valid
    np_labels = labels.cpu().data.numpy()

    _, _, all_acc = base_model(features, Variable(base_model.norm_tool.normed_adj), idx_test, labels)
    all_acc = all_acc.cpu().numpy()
    print('acc before modification:', np.mean(all_acc))
    
    attacked = 0.0
    pbar = tqdm(range(len(idx_test)))
    for pos in pbar:
        if all_acc[pos] < 1.0:
            attacked += 1
            continue
        idx = idx_test[pos]
        mod = ModifiedGraph()
        for i in range(cmd_args.num_mod):
            x = None
            y = None
            if len(set(khop_neighbors[idx])) == 0:
                continue
            if len(set(khop_neighbors[idx])) == 1 and idx in khop_neighbors[idx]:
                continue
            while True:
                region = khop_neighbors[idx]
                x = region[np.random.randint(len(region))]                                
                region = khop_neighbors[x]                
                y = region[np.random.randint(len(region))]
                
                if x == y:
                    continue
                assert x in khop_neighbors[y] and y in khop_neighbors[x]
                break
            if x is not None and y is not None:
                mod.add_edge(x, y, -1.0)
        if len(mod.directed_edges) != cmd_args.num_mod:
            continue
        adj = base_model.norm_tool.norm_extra(mod.get_extra_adj())
        _, _, acc = base_model(features, Variable(adj), [idx], labels)
        acc = acc.double().cpu().numpy()
        if acc[0] < 1.0:
            attacked += 1            

        pbar.set_description('cur_attack: %.2f' % (attacked) )

    print( '%.6f\n' % (1.0 - attacked / len(idx_test)) )

    