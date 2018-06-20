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

sys.path.append('%s/../../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from graph_embedding import gnn_spmm

sys.path.append('%s/../node_classification' % os.path.dirname(os.path.realpath(__file__)))
from node_utils import GraphNormTool

def node_greedy_actions(target_nodes, picked_nodes, list_q, net):
    assert len(target_nodes) == len(list_q)

    actions = []
    values = []
    for i in range(len(target_nodes)):
        region = net.list_action_space[target_nodes[i]]
        if picked_nodes is not None and picked_nodes[i] is not None:
            region = net.list_action_space[picked_nodes[i]]
        if region is None:
            assert list_q[i].size()[0] == net.total_nodes
        else:
            assert len(region) == list_q[i].size()[0]
        
        val, act = torch.max(list_q[i], dim=0)
        values.append(val)
        if region is not None:
            act = region[act.data.cpu().numpy()[0]]
            act = Variable(torch.LongTensor([act]))
            actions.append( act )
        else:
            actions.append(act)

    return torch.cat(actions, dim=0).data, torch.cat(values, dim=0).data

class QNetNode(nn.Module):
    def __init__(self, node_features, node_labels, list_action_space):
        super(QNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        embed_dim = cmd_args.latent_dim
        if cmd_args.bilin_q:
            last_wout = embed_dim
        else:
            last_wout = 1
            self.bias_target = Parameter(torch.Tensor(1, embed_dim))

        if cmd_args.mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, cmd_args.mlp_hidden)
            self.linear_out = nn.Linear(cmd_args.mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)

        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(cmd_args.adj_norm, cmd_args.gm)
        weights_init(self)

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)

        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows, n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp.cuda()
        return sp

    def forward(self, time_t, states, actions, greedy_acts = False, is_inference=False):
        if self.node_features.data.is_sparse:
            input_node_linear = gnn_spmm(self.node_features, self.w_n2l)            
        else:
            input_node_linear = torch.mm(self.node_features, self.w_n2l)
        input_node_linear += self.bias_n2l

        target_nodes, batch_graph, picked_nodes = zip(*states)

        list_pred = []
        prefix_sum = []
        for i in range(len(batch_graph)):
            region = self.list_action_space[target_nodes[i]]

            node_embed = input_node_linear.clone()
            if picked_nodes is not None and picked_nodes[i] is not None:
                picked_sp = Variable( self.make_spmat(self.total_nodes, 1, picked_nodes[i], 0), volatile=is_inference )
                node_embed += gnn_spmm(picked_sp, self.bias_picked)
                region = self.list_action_space[picked_nodes[i]]

            if not cmd_args.bilin_q:
                target_sp = Variable( self.make_spmat(self.total_nodes, 1, target_nodes[i], 0), volatile=is_inference)
                node_embed += gnn_spmm(target_sp, self.bias_target)

            adj = Variable( self.norm_tool.norm_extra( batch_graph[i].get_extra_adj() ), volatile=is_inference )
            lv = 0
            input_message = node_embed
            node_embed = F.relu(input_message)
            while lv < cmd_args.max_lv:
                n2npool = gnn_spmm(adj, node_embed)
                node_linear = self.conv_params( n2npool )
                merged_linear = node_linear + input_message
                node_embed = F.relu(merged_linear)
                lv += 1

            target_embed = node_embed[target_nodes[i], :].view(-1, 1)
            if region is not None:
                node_embed = node_embed[region]

            graph_embed = torch.mean(node_embed, dim=0, keepdim=True)
            
            if actions is None:
                graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
            else:
                if region is not None:
                    act_idx = region.index(actions[i])
                else:
                    act_idx = actions[i]
                node_embed = node_embed[act_idx, :].view(1, -1)

            embed_s_a = torch.cat((node_embed, graph_embed), dim=1)
            if cmd_args.mlp_hidden:
                embed_s_a = F.relu( self.linear_1(embed_s_a) )            
            raw_pred = self.linear_out(embed_s_a)

            if cmd_args.bilin_q:
                raw_pred = torch.mm(raw_pred, target_embed)
            list_pred.append(raw_pred)

        if greedy_acts: 
            actions, _ = node_greedy_actions(target_nodes, picked_nodes, list_pred, self)

        return actions, list_pred

class NStepQNetNode(nn.Module):
    def __init__(self, num_steps, node_features, node_labels, list_action_space):
        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)

        list_mod = []

        for i in range(0, num_steps):
            list_mod.append(QNetNode(node_features, node_labels, list_action_space))
        
        self.list_mod = nn.ModuleList(list_mod)

        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts = False, is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_mod[time_t](time_t, states, actions, greedy_acts, is_inference)
        