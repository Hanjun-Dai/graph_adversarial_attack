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

def propose_add(grad):
    idxes = np.argsort(grad)
    added = []

    mod = ModifiedGraph()
    for p in idxes:
        x = p // len(StaticGraph.graph)
        y = p % len(StaticGraph.graph)
        if x == y or x in dict_of_lists[y] or y in dict_of_lists[x]:
            continue
        if cmd_args.n_hops > 0 and not x in khop_neighbors[y]:
            continue
        assert cmd_args.n_hops <= 0 or (x in khop_neighbors[y] and y in khop_neighbors[x])
        mod.add_edge(x, y, 1.0)
        if len(mod.directed_edges) >= cmd_args.num_mod:
            break
    if len(mod.directed_edges) < cmd_args.num_mod:
        extra = None
    else:
        extra = mod.get_extra_adj()    
    adj = base_model.norm_tool.norm_extra(extra)
    _, _, acc = base_model(features, Variable(adj), [idx], labels)
    acc = acc.double().cpu().numpy()

    return acc[0] < 1.0, mod

def propose_del(grad):
    idxes = np.argsort(-grad)
    added = []

    mod = ModifiedGraph()
    for p in idxes:
        x = p // len(StaticGraph.graph)
        y = p % len(StaticGraph.graph)
        if x == y:
            continue
        if not x in dict_of_lists[y] or not y in dict_of_lists[x]:
            continue        
        mod.add_edge(x, y, -1.0)
        if len(mod.directed_edges) >= cmd_args.num_mod:
            break
    if len(mod.directed_edges) < cmd_args.num_mod:
        extra = None
    else:
        extra = mod.get_extra_adj()
    adj = base_model.norm_tool.norm_extra(extra)
    pred, _, acc = base_model(features, Variable(adj), [idx], labels)
    acc = acc.double().cpu().numpy()

    return acc[0] < 1.0, mod, pred.cpu().numpy()


if __name__ == '__main__':    
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    
    features, labels, _, idx_test, base_model, khop_neighbors = init_setup()
    np_labels = labels.cpu().data.numpy()

    method = propose_del
    attacked = 0.0
    pbar = tqdm(range(len(idx_test)))
        
    dict_of_lists = load_raw_graph(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)
    with open('%s/%s-grad-labels.txt' % (cmd_args.save_dir, cmd_args.dataset), 'w', 0) as f:
        for i in range(len(np_labels)):
            f.write('%d\n' % np_labels[i])

    _, _, all_acc = base_model(features, Variable(base_model.norm_tool.normed_adj), idx_test, labels)
    all_acc = all_acc.cpu().numpy()
    fsol = open('%s/%s-grad.txt' % (cmd_args.save_dir, cmd_args.dataset), 'w', 0)
    for pos in pbar:
        if all_acc[pos] < 1.0:
            attacked += 1
            continue
        idx = idx_test[pos]
        fake_labels = labels.clone()

        if cmd_args.targeted:
            for i in range(cmd_args.num_class):
                if i == np_labels[idx]:
                    continue
                adj = Variable( base_model.norm_tool.normed_adj, requires_grad=True )
                base_model.zero_grad()
                fake_labels[idx] = i
                _, loss, acc = base_model(features, adj, [idx], fake_labels)
                loss.backward()
                grad = adj.grad.data.cpu().numpy().flatten()

                if method(grad)[0]:
                    attacked += 1
                    break
        else:
            adj = Variable( base_model.norm_tool.normed_adj, requires_grad=True )
            base_model.zero_grad()            
            _, loss, acc = base_model(features, adj, [idx], labels)
            loss = -loss
            loss.backward()
            grad = adj.grad.data.cpu().numpy().flatten()
            succ, mod, pred = method(grad)
            if succ:
                fsol.write('%d: [%d %d]\n' % (idx, mod.directed_edges[0][0], mod.directed_edges[0][1]))
                ftxt = open('%s/%s-grad-%d.txt' % (cmd_args.save_dir, cmd_args.dataset, idx), 'w', 0)
                ftxt.write('origin: %d, pred: %d\n' % (np_labels[idx], pred[0]))
                seen = set()
                for i in dict_of_lists[idx]:
                    for j in dict_of_lists[i]:
                        if (i, j) in seen or (j, i) in seen:
                            continue
                        score = grad[i * len(StaticGraph.graph) + j]
                        ftxt.write('%d %d %.4f\n' % (i, j, score))
                        score = grad[j * len(StaticGraph.graph) + i]
                        ftxt.write('%d %d %.4f\n' % (j, i, score))

                attacked += 1
                ftxt.close()

        pbar.set_description('cur_attack: %.2f' % (attacked) )
    fsol.close()
    print( '%.6f\n' % (1.0 - attacked / len(idx_test)) )

    
