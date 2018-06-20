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

sys.path.append('%s/../../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import gnn_spmm

from node_utils import GraphNormTool

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        weights_init(self)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = gnn_spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        
        output = gnn_spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNModule(nn.Module):
    def __init__(self, **kwargs):
        super(GCNModule, self).__init__()
        self.gc1 = GraphConvolution(kwargs['feature_dim'], kwargs['latent_dim'])
        self.gc2 = GraphConvolution(kwargs['latent_dim'], kwargs['num_class'])
        self.dropout_rate = kwargs['dropout']
        self.norm_tool = GraphNormTool(kwargs['adj_norm'], 'gcn')

    def forward(self, x, adj, node_selector = None, labels = None, avg_loss = True):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj)        
        logits = F.log_softmax(x, dim=1)

        if node_selector is not None:
            logits = logits[node_selector]
        
        if labels is not None:
            if node_selector is not None:
                labels = labels[node_selector]
            loss = F.nll_loss(logits, labels, reduce=avg_loss)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(labels.data.view_as(pred)).cpu()
            return pred, loss, acc
        else:
            return pred

class S2VNodeClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(S2VNodeClassifier, self).__init__()        
        self.w_n2l = Parameter(torch.Tensor(kwargs['feature_dim'], kwargs['latent_dim']))
        self.bias_n2l = Parameter(torch.Tensor(kwargs['latent_dim']))
        self.conv_params = nn.Linear(kwargs['latent_dim'], kwargs['latent_dim'])        
        self.max_lv = kwargs['max_lv']
        self.dropout_rate = kwargs['dropout']
        self.last_weights = nn.Linear(kwargs['latent_dim'], kwargs['num_class'])
        self.norm_tool = GraphNormTool(kwargs['adj_norm'], 'mean_field')
        
        weights_init(self)

    def forward(self, x, adj, node_selector = None, labels = None, avg_loss = True):
        if x.data.is_sparse:
            input_node_linear = gnn_spmm(x, self.w_n2l)
        else:
            input_node_linear = torch.mm(x, self.w_n2l)
        input_node_linear += self.bias_n2l
        
        # n2npool = gnn_spmm(adj, input_node_linear)
        # cur_message_layer = F.relu(n2npool)
        # cur_message_layer = F.dropout(cur_message_layer, self.dropout_rate, training=self.training)

        # node_embed = gnn_spmm(adj, cur_message_layer)

        input_potential = F.relu(input_node_linear)
        lv = 0
        node_embed = input_potential
        while lv < self.max_lv:
            n2npool = gnn_spmm(adj, node_embed)
            node_linear = self.conv_params( n2npool )
            merged_linear = node_linear + input_node_linear
            node_embed = F.relu(merged_linear) + node_embed
            lv += 1

        logits = self.last_weights(node_embed)
        logits = F.log_softmax(logits, dim=1)

        if node_selector is not None:
            logits = logits[node_selector]
        
        if labels is not None:
            if node_selector is not None:
                labels = labels[node_selector]
            loss = F.nll_loss(logits, labels, reduce=avg_loss)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(labels.data.view_as(pred)).cpu()
            return pred, loss, acc
        else:
            return pred