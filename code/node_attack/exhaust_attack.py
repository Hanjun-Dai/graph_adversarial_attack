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
import cPickle as cp
import time

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../node_classification' % os.path.dirname(os.path.realpath(__file__)))
from node_utils import load_txt_data, load_binary_data, run_test, load_raw_graph

from node_attack_common import load_base_model, ModifiedGraph

def check_attack_rate(gcn, features, labels, idx_test, list_of_modification):
    all_acc = torch.ones(len(idx_test), 1)
    pbar = tqdm(list_of_modification)
    _, _, orig_acc = gcn(features, Variable(gcn.norm_tool.normed_adj), idx_test, labels)
    attackable = {}
    ftxt = open('%s/%s-exaust.txt' % (cmd_args.save_dir, cmd_args.dataset), 'w', 0)

    for g in pbar:
        adj = gcn.norm_tool.norm_extra(g.get_extra_adj())
        _, _, acc = gcn(features, Variable(adj), idx_test, labels)
        
        for i in range(len(idx_test)):
            if float(acc[i]) < float(orig_acc[i]):
                if not idx_test[i] in attackable:
                    attackable[idx_test[i]] = []
                attackable[idx_test[i]].append(g.directed_edges)
                ftxt.write('%d:' %idx_test[i])
                for e in g.directed_edges:
                    ftxt.write(' %d %d' % (e[0], e[1]))
                ftxt.write('\n')
        all_acc *= acc.float()

        cur_acc = all_acc.sum() / float(len(idx_test))

        pbar.set_description('cur_acc: %0.5f' % (cur_acc) )
    with open('%s/%s-exaust.pkl' % (cmd_args.save_dir, cmd_args.dataset), 'wb') as f:
        cp.dump(attackable, f, cp.HIGHEST_PROTOCOL)

def gen_modified(dict_of_lists, mod_type):
    for i in range(len(dict_of_lists)):
        if mod_type == 'any' or mod_type == 'del':
            for j in dict_of_lists[i]:
                yield ModifiedGraph([(i, j)], [-1.0])
        if mod_type == 'del':
            continue
        for j in range(i + 1, len(dict_of_lists)):
            if not j in dict_of_lists[i]:
                g = ModifiedGraph([(i, j)], [1.0])
                yield g

def recur_gen_edges(center, khop_neighbors, dict_of_lists, cur_list, n_edges):        
    for j in khop_neighbors[center]:
        if not j in dict_of_lists[center] and j != center:
            new_list = cur_list + [(center, j)]            
            if len(new_list) == n_edges:                
                g = ModifiedGraph(new_list, [1.0] * n_edges)
                yield g
            else:                
                for g in recur_gen_edges(center, khop_neighbors, dict_of_lists, new_list, n_edges):
                    yield g

def gen_khop_edges(khop_neighbors, dict_of_lists, n_edges):
    for i in range(len(dict_of_lists)):
        for g in recur_gen_edges(i, khop_neighbors, dict_of_lists, [], n_edges):
            yield g

if __name__ == '__main__':    
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    features, labels, idx_train, idx_val, idx_test = load_txt_data(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)
    if cmd_args.meta_test:
        idx_test = idx_val

    features = Variable( features )
    labels = Variable( torch.LongTensor( np.argmax(labels, axis=1) ) )
    if cmd_args.ctx == 'gpu':
        labels = labels.cuda()
    
    base_model = load_base_model()
    run_test(base_model, features, Variable( base_model.norm_tool.normed_adj ), idx_test, labels)
    
    dict_of_lists = load_raw_graph(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)

    # freely add edges
    if cmd_args.n_hops <= 0:
        check_attack_rate(base_model, features, labels, idx_test, gen_modified(dict_of_lists, 'del'))
    else:
        # add edges within k-hop
        pass
        # khop_neighbors = get_khop_neighbors(dict_of_lists, cmd_args.n_hops)
        # nei_size = []
        # for i in khop_neighbors:
        #     nei_size.append(len(khop_neighbors[i]))
        # print(np.mean(nei_size), np.max(nei_size), np.min(nei_size))
        # check_attack_rate(base_model, features, labels, idx_test, gen_khop_edges(khop_neighbors, dict_of_lists, 1))
