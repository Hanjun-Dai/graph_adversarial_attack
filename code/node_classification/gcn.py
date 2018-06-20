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

sys.path.append('%s/../../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args, save_args
from gcn_modules import GCNModule, S2VNodeClassifier
from node_utils import load_binary_data, load_txt_data, run_test, StaticGraph

def adj_generator():
    directed_edges = StaticGraph.graph.edges()

    while True:
        if cmd_args.del_rate > 0:
            random.shuffle(directed_edges)
            del_num = int(len(directed_edges) * cmd_args.del_rate)
            for i in range(len(directed_edges) // del_num):
                cur_edges = directed_edges[i * del_num : (i + 1) * del_num]
                edges = np.array(cur_edges, dtype=np.int64)
                rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
                edges = np.hstack((edges.T, rev_edges))
                idxes = torch.LongTensor(edges)
                values = torch.ones(idxes.size()[1]) * -1.0

                added = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
                if cmd_args.ctx == 'gpu':
                    added = added.cuda()
                
                new_adj = gcn.norm_tool.norm_extra(added)
                yield Variable(new_adj)
        else:
            yield orig_adj

if __name__ == '__main__':    
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)    

    # features, labels, idx_train, idx_val, idx_test = load_binary_data(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)
    features, labels, idx_train, idx_val, idx_test = load_txt_data(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)    
    features = Variable( features )
    labels = Variable( torch.LongTensor( np.argmax(labels, axis=1) ) )

    if cmd_args.gm == 'mean_field':
        mod = S2VNodeClassifier
    elif cmd_args.gm == 'gcn':
        mod = GCNModule
    if cmd_args.saved_model is not None:
        print('loading')
        with open('%s-args.pkl' % cmd_args.saved_model, 'rb') as f:
            base_args = cp.load(f)
        gcn = mod(**vars(base_args))
        gcn.load_state_dict(torch.load(cmd_args.saved_model+ '.model'))        
    else:
        gcn = mod(**vars(cmd_args))

    orig_adj = Variable( gcn.norm_tool.normed_adj )

    if cmd_args.ctx == 'gpu':
        gcn = gcn.cuda()
        labels = labels.cuda()

    if cmd_args.phase == 'test':
        run_test(gcn, features, orig_adj, idx_test, labels)
        sys.exit()

    optimizer = optim.Adam(gcn.parameters(), lr=cmd_args.learning_rate, weight_decay=cmd_args.weight_decay)
    best_val = None
    gen = adj_generator()
    for epoch in range(cmd_args.num_epochs):
        t = time.time()
        gcn.train()
        optimizer.zero_grad()
        cur_adj = next(gen)
        _, loss_train, acc_train = gcn(features, cur_adj, idx_train, labels)
        acc_train = acc_train.sum() / float(len(idx_train))
        loss_train.backward()
        optimizer.step()

        gcn.eval()
        _, loss_val, acc_val = gcn(features, orig_adj, idx_val, labels)
        acc_val = acc_val.sum() / float(len(idx_val))

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data[0]),
            'acc_train: {:.4f}'.format(acc_train),
            'loss_val: {:.4f}'.format(loss_val.data[0]),
            'acc_val: {:.4f}'.format(acc_val),
            'time: {:.4f}s'.format(time.time() - t))
        
        if best_val is None or acc_val > best_val:
            best_val = acc_val
            print('----saving to best model since this is the best valid loss so far.----')
            torch.save(gcn.state_dict(), cmd_args.save_dir + '/model-%s-epoch-best-%.2f.model' % (cmd_args.gm, cmd_args.del_rate))
            save_args(cmd_args.save_dir + '/model-%s-epoch-best-%.2f-args.pkl' % (cmd_args.gm, cmd_args.del_rate), cmd_args)

    run_test(gcn, features, orig_adj, idx_test, labels)
    # pred = gcn(features, adh)
