import numpy as np
import pickle as pkl
import networkx as nx
import torch
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os

sys.path.append('%s/../common/functions' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from functions.custom_func import GraphLaplacianNorm, GraphDegreeNorm
from cmd_args import cmd_args

class StaticGraph(object):
    graph = None

    @staticmethod
    def get_gsize():
        return torch.Size( (len(StaticGraph.graph), len(StaticGraph.graph)) )

class GraphNormTool(object):
    def __init__(self, adj_norm, gm):
        self.adj_norm = adj_norm
        self.gm = gm
        g = StaticGraph.graph

        edges = np.array(g.edges(), dtype=np.int64)
        rev_edges = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
        self_edges = np.array([range(len(g)), range(len(g))], dtype=np.int64)

        edges = np.hstack((edges.T, rev_edges, self_edges))

        idxes = torch.LongTensor(edges)
        values = torch.ones(idxes.size()[1])

        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
        if cmd_args.ctx == 'gpu':
            self.raw_adj = self.raw_adj.cuda()
        
        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                GraphLaplacianNorm(self.normed_adj)
            else:
                GraphDegreeNorm(self.normed_adj)

    def norm_extra(self, added_adj = None):
        if added_adj is None:
            return self.normed_adj

        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                GraphLaplacianNorm(new_adj)
            else:
                GraphDegreeNorm(new_adj)
        return new_adj

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_raw_graph(data_folder, dataset_str):
    bin_file = "{}/ind.{}.{}".format(data_folder, dataset_str, 'graph')
    if os.path.isfile(bin_file):
        with open(bin_file, 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)
    else:
        txt_file = data_folder + '/adj_list.txt'
        graph = {}
        with open(txt_file, 'r') as f:
            cur_idx = 0
            for row in f:
                row = row.strip().split()
                adjs = []
                for j in range(1, len(row)):
                    adjs.append(int(row[j]))
                graph[cur_idx] = adjs
                cur_idx += 1

    return graph

def load_binary_data(data_folder, dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(data_folder, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(data_folder, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    StaticGraph.graph = nx.from_dict_of_lists(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    cmd_args.feature_dim = features.shape[1]
    cmd_args.num_class = labels.shape[1]
    return preprocess_features(features), labels, idx_train, idx_val, idx_test

def load_txt_data(data_folder, dataset_str):
    idx_train = list(np.loadtxt(data_folder + '/train_idx.txt', dtype=int))
    idx_val = list(np.loadtxt(data_folder + '/val_idx.txt', dtype=int))
    idx_test = list(np.loadtxt(data_folder + '/test_idx.txt', dtype=int))
    labels = np.loadtxt(data_folder + '/label.txt')
    
    with open(data_folder + '/meta.txt', 'r') as f:
        num_nodes, cmd_args.num_class, cmd_args.feature_dim = [int(w) for w in f.readline().strip().split()]

    graph = load_raw_graph(data_folder, dataset_str)
    assert len(graph) == num_nodes
    StaticGraph.graph = nx.from_dict_of_lists(graph)
    
    row_ptr = []
    col_idx = []
    vals = []
    with open(data_folder + '/features.txt', 'r') as f:
        nnz = 0
        for row in f:
            row = row.strip().split()
            row_ptr.append(nnz)            
            for i in range(1, len(row)):
                w = row[i].split(':')
                col_idx.append(int(w[0]))
                vals.append(float(w[1]))
            nnz += int(row[0])
        row_ptr.append(nnz)
    assert len(col_idx) == len(vals) and len(vals) == nnz and len(row_ptr) == num_nodes + 1

    features = sp.csr_matrix((vals, col_idx, row_ptr), shape=(num_nodes, cmd_args.feature_dim))
    
    return preprocess_features(features), labels, idx_train, idx_val, idx_test



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)    
    sp_tuple = sparse_to_tuple(features)
    idxes = torch.LongTensor(sp_tuple[0]).transpose(0, 1).contiguous()
    values = torch.Tensor(sp_tuple[1].astype(np.float32))

    mat = torch.sparse.FloatTensor(idxes, values, torch.Size(sp_tuple[2]))
    if cmd_args.ctx == 'gpu':
        mat = mat.cuda()
    return mat

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def run_test(gcn, features, adj, idx_test, labels):
    gcn.eval()
    _, loss_test, acc_test = gcn(features, adj, idx_test, labels)
    acc_test = acc_test.sum() / float(len(idx_test))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test))
