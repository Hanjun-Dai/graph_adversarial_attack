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
import cPickle as cp

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import S2VGraph
from cmd_args import cmd_args

sys.path.append('%s/../node_classification' % os.path.dirname(os.path.realpath(__file__)))
from node_utils import load_txt_data, run_test, load_raw_graph, StaticGraph
from gcn_modules import GCNModule, S2VNodeClassifier

class ModifiedGraph(object):
    def __init__(self, directed_edges = None, weights = None):
        if directed_edges is not None:
            self.directed_edges = deepcopy(directed_edges)
            self.weights = deepcopy(weights)
        else:
            self.directed_edges = []
            self.weights = []

    def add_edge(self, x, y, z):
        assert x is not None and y is not None
        if x == y:
            return
        for e in self.directed_edges:
            if e[0] == x and e[1] == y:
                return
            if e[1] == x and e[0] == y:
                return
        self.directed_edges.append((x, y))
        assert z < 0
        self.weights.append(-1.0)

    def get_extra_adj(self):
        if len(self.directed_edges):
            edges = np.array(self.directed_edges, dtype=np.int64)
            rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
            edges = np.hstack((edges.T, rev_edges))

            idxes = torch.LongTensor(edges)
            values = torch.Tensor(self.weights + self.weights)
            
            added_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
            if cmd_args.ctx == 'gpu':
                added_adj = added_adj.cuda()
            return added_adj
        else:
            return None

class NodeAttakEnv(object):
    def __init__(self, features, labels, all_targets, list_action_space, classifier):
        self.classifier = classifier
        self.list_action_space = list_action_space
        self.features = features
        self.labels = labels
        self.all_targets = all_targets

    def setup(self, target_nodes):
        self.target_nodes = target_nodes
        self.n_steps = 0
        self.first_nodes = None
        self.rewards = None
        self.binary_rewards = None
        self.modified_list = []
        for i in range(len(self.target_nodes)):
            self.modified_list.append(ModifiedGraph())

        self.list_acc_of_all = []

    def step(self, actions):
        if self.first_nodes is None: # pick the first node of edge
            assert self.n_steps % 2 == 0
            self.first_nodes = actions[:]
        else:
            for i in range(len(self.target_nodes)):
                #assert self.first_nodes[i] != actions[i]
                self.modified_list[i].add_edge(self.first_nodes[i], actions[i], -1.0)
            self.first_nodes = None
            self.banned_list = None
        self.n_steps += 1

        if self.isTerminal():
            acc_list = []
            loss_list = []
            for i in tqdm(range(len(self.target_nodes))):
                extra_adj = self.modified_list[i].get_extra_adj()
                adj = self.classifier.norm_tool.norm_extra(extra_adj)
                _, loss, acc = self.classifier(self.features, Variable(adj), self.all_targets, self.labels, avg_loss=False)
                cur_idx = self.all_targets.index(self.target_nodes[i])
                acc = np.copy(acc.double().cpu().view(-1).numpy())
                loss = loss.data.cpu().view(-1).numpy()
                self.list_acc_of_all.append(acc)
                acc_list.append(acc[cur_idx])
                loss_list.append(loss[cur_idx])
            self.binary_rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            if cmd_args.reward_type == 'binary': 
                self.rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
            else:
                assert cmd_args.reward_type == 'nll'
                self.rewards = np.array(loss_list).astype(np.float32)

    def sample_pos_rewards(self, num_samples):
        assert self.list_acc_of_all is not None
        cands = []
        for i in range(len(self.list_acc_of_all)):
            succ = np.where( self.list_acc_of_all[i] < 0.9 )[0]
            for j in range(len(succ)):
                cands.append((i, self.all_targets[succ[j]]))
        if num_samples > len(cands):
            return cands
        random.shuffle(cands)
        return cands[0:num_samples]

    def uniformRandActions(self):
        act_list = []
        offset = 0
        for i in range(len(self.target_nodes)):
            cur_node = self.target_nodes[i]
            region = self.list_action_space[cur_node]
            
            if self.first_nodes is not None and self.first_nodes[i] is not None:
                region = self.list_action_space[self.first_nodes[i]]

            if region is None:
                cur_action = np.random.randint(len(self.list_action_space))
            else:                
                cur_action = region[np.random.randint(len(region))]

            act_list.append(cur_action)
        return act_list

    def isTerminal(self):
        if self.n_steps == 2 * cmd_args.num_mod:
            return True
        return False
    
    def getStateRef(self):
        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes

        return zip(self.target_nodes, self.modified_list, cp_first)
                        
    def cloneState(self):
        cp_first = [None] * len(self.target_nodes)
        if self.first_nodes is not None:
            cp_first = self.first_nodes[:]
                
        return zip(self.target_nodes[:], deepcopy(self.modified_list), cp_first)
        
def load_base_model():
    assert cmd_args.saved_model is not None
    with open('%s-args.pkl' % cmd_args.saved_model, 'rb') as f:
        base_args = cp.load(f)
    if 'mean_field' in cmd_args.saved_model:
        mod = S2VNodeClassifier
    elif 'gcn' in cmd_args.saved_model:
        mod = GCNModule

    gcn = mod(**vars(base_args))
    if cmd_args.ctx == 'gpu':
        gcn = gcn.cuda()
    gcn.load_state_dict(torch.load(cmd_args.saved_model+ '.model'))
    gcn.eval()
    return gcn

def init_setup():
    features, labels, _, idx_val, idx_test = load_txt_data(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)    
    features = Variable( features )
    labels = Variable( torch.LongTensor( np.argmax(labels, axis=1) ) )
    if cmd_args.ctx == 'gpu':
        labels = labels.cuda()

    base_model = load_base_model()
    run_test(base_model, features, Variable( base_model.norm_tool.normed_adj ), idx_test, labels)

    dict_of_lists = load_raw_graph(cmd_args.data_folder + '/' + cmd_args.dataset, cmd_args.dataset)

    return features, labels, idx_val, idx_test, base_model, dict_of_lists
