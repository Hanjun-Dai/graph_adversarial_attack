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

from node_attack_common import ModifiedGraph, init_setup

class NodeGeneticAgent(object):
    def __init__(self, features, labels, list_action_space, classifier, n_edges_attack, target_node):
        self.n_edges_attack = n_edges_attack
        self.classifier = classifier
        self.list_action_space = list_action_space
        self.features = features
        self.labels = labels
        self.target_node = target_node
        self.total_nodes = len(self.list_action_space)
        self.solution = None
        self.population = []

        region = self.list_action_space[target_node]
        if len(set(region)) == 0:
            return
        if len(set(region)) == 1 and self.target_node in region:
            return

        for k in range(cmd_args.population_size):
            added = ModifiedGraph()
            for k in range(n_edges_attack):
                while True:
                    x = self.rand_action(self.target_node)
                    y = self.rand_action(x)
                    if x == y:
                        continue
                    break
                added.add_edge(x, y, -1.0)
            self.population.append(added)

    def rand_action(self, x):
        region = self.list_action_space[x]
        y = region[np.random.randint(len(region))]
        return y
        
    def get_fitness(self):
        nll_list = []
        for i in range(len(self.population)):
            adj = self.classifier.norm_tool.norm_extra( self.population[i].get_extra_adj() )
            adj = Variable(adj, volatile=True)
            _, loss, acc = self.classifier(self.features, adj, [self.target_node], self.labels)
            nll_list.append(loss.cpu().data.numpy()[0])
            # print(i, self.population[i].directed_edges, float(acc.cpu()[0]))
            if self.solution is None and float(acc.cpu()[0]) < 1.0: # successed
                self.solution = self.population[i]
                break
        return np.array(nll_list)

    def select(self, fitness):                
        scores = np.exp(fitness)
        max_args = np.argsort(-scores)

        result = []
        for i in range(cmd_args.population_size - cmd_args.population_size // 2):
            result.append(deepcopy(self.population[max_args[i]]))

        idx = np.random.choice(np.arange(cmd_args.population_size), 
                                size=cmd_args.population_size // 2,
                                replace=True, 
                                p=scores/scores.sum())
        for i in idx:
            result.append(deepcopy(self.population[i]))                                
        return result

    def crossover(self, parent, pop):
        if np.random.rand() < cmd_args.cross_rate:
            another = pop[ np.random.randint(len(pop)) ]
            if len(parent.directed_edges) == 0:
                return deepcopy(another)
            if len(another.directed_edges) == 0:
                return deepcopy(parent)
            new_graph = ModifiedGraph()
            for i in range(self.n_edges_attack):
                if np.random.rand() < 0.5:
                    e = parent.directed_edges[i]
                    new_graph.add_edge(e[0], e[1], parent.weights[i])
                else:
                    e = another.directed_edges[i]
                    new_graph.add_edge(e[0], e[1], another.weights[i])                    
            return new_graph
        else:
            return deepcopy(parent)

    def mutate(self, child):
        for i in range(self.n_edges_attack):
            if len(child.directed_edges) == 0:
                continue
            if np.random.rand() < cmd_args.mutate_rate:
                if np.random.rand() < 0.5:
                    new_e = (child.directed_edges[i][0], self.rand_action(child.directed_edges[i][0]))
                    child.directed_edges[i] = new_e
                else:
                    new_e = (self.rand_action(child.directed_edges[i][1]), child.directed_edges[i][1])
                    child.directed_edges[i] = new_e           

    def evolve(self):
        fitness = self.get_fitness()
        if self.solution is not None:
            return
        pop = self.select(fitness)
        new_pop_list = []
        for parent in pop:
            child = self.crossover(parent, pop)
            self.mutate(child)
            new_pop_list.append(child)

        self.population = new_pop_list        

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    features, labels, _, idx_test, base_model, khop_neighbors = init_setup()

    if cmd_args.idx_start + cmd_args.num_instances > len(idx_test):
        instances = idx_test[cmd_args.idx_start : ]
    else:
        instances = idx_test[cmd_args.idx_start : cmd_args.idx_start + cmd_args.num_instances]

    pbar = tqdm(instances)
    attacked = 0.0
    for g in pbar:
        agent = NodeGeneticAgent(features, labels, khop_neighbors, base_model, cmd_args.num_mod, g)
        if len(agent.population) == 0:
            continue
        for i in range(cmd_args.rounds):
            agent.evolve()
            if agent.solution is not None:
                attacked += 1
                break
        with open('%s/sol-%d.txt' % (cmd_args.save_dir, g), 'w') as f:
            f.write('%d: [' % g)
            if agent.solution is not None:
                for e in agent.solution.directed_edges:
                    f.write('(%d, %d)' % e)
            f.write('] succ: ')
            if agent.solution is not None:
                f.write('1\n')
            else:
                f.write('0\n')
        pbar.set_description('cur_attack: %.2f' % (attacked) )

    print('\n\nacc: %.4f\n' % ((len(instances) - attacked) / float(len(instances))) )
