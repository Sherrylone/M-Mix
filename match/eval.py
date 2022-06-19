from dataset import load
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from model import Model
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import graphkernels.kernels as gk
import igraph as ig
from sklearn.metrics import roc_auc_score

def generate_one_graph(adj, feat):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(adj.shape[-1])])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]!=0:
                G.add_edge(i, j, weight=adj[i, j])
    return G

def generate_graph(adjs, feat):
    num_graph = adjs.shape[0]
    adj_pos, adj_neg, diff_pos, diff_neg = [], [], [], []
    graphs_pos = []
    graphs_negs = []
    graphs = []
    for i in tqdm(range(num_graph)):
        graph = generate_one_graph(adjs[i], feat[i])
        graph_pos = substitute_random_edges(graph, 1)
        graph_neg = substitute_random_edges(graph, 2)
        graphs.append(graph)
        graphs_pos.append(graph_pos)
        graphs_negs.append(graph_neg)
        adj_pos.append(np.array(nx.to_numpy_array(graph_pos)))
        adj_neg.append(np.array(nx.to_numpy_array(graph_neg)))
        diff_pos.append(compute_pp(np.array(nx.to_numpy_array(graph_pos))))
        diff_neg.append(compute_pp(np.array(nx.to_numpy_array(graph_neg))))
    return np.array(adj_pos), np.array(adj_neg), np.array(diff_pos), np.array(diff_neg), graphs, graphs_pos, graphs_negs

def substitute_random_edges(g, n):
    """Substitutes n edges from graph g with another n randomly picked edges."""
    g = copy.deepcopy(g)
    n_nodes = g.number_of_nodes()
    edges = list(g.edges())
    # sample n edges without replacement
    e_remove = [
        edges[i] for i in np.random.choice(np.arange(len(edges)), n, replace=False)
    ]
    edge_set = set(edges)
    e_add = set()
    while len(e_add) < n:
        e = np.random.choice(n_nodes, 2, replace=False)
        # make sure e does not exist and is not already chosen to be added
        if (
                (e[0], e[1]) not in edge_set
                and (e[1], e[0]) not in edge_set
                and (e[0], e[1]) not in e_add
                and (e[1], e[0]) not in e_add
        ):
            e_add.add((e[0], e[1]))

    for i, j in e_remove:
        g.remove_edge(i, j)
    for i, j in e_add:
        g.add_edge(i, j)
    return g

def compute_pp(a, alpha=0.2, self_loop=True):
    a = np.sign(a)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def generate_ig_graph(adj, feat=None):
    G = ig.Graph()
    for i in range(adj.shape[-1]):
        G.add_vertices(i)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]!=0:
                G.add_edges([(i, j)])
    return G

def transform(graphs):
    out = []
    for graph in graphs:
        adj = np.array(nx.to_numpy_array(graph))
        new = generate_ig_graph(adj)
        out.append(new)
    return out

def compute_auc(pre, gt):
    return roc_auc_score(gt, pre)

def cal_kernel(graphs, graphs1, graphs2, kernel='WL'):
    graphs = graphs1 + graphs + graphs2
    if kernel == "WL":
        k = gk.CalculateWLKernel(graphs, par=5)
    elif kernel == 'edge-hist':
        k = gk.CalculateEdgeHistKernel(graphs)
    elif kernel == 'vertex-hist':
        k = gk.CalculateVertexHistKernel(graphs)
    elif kernel == 'vertex-edge-hist':
        k = gk.CalculateVertexEdgeHistKernel(graphs)
    elif kernel == 'random-walk':
        k = gk.CalculateGeometricRandomWalkKernel(graphs)
    elif kernel == 'e-random-walk':
        k = gk.CalculateExponentialRandomWalkKernel(graphs)
    elif kernel == 'short-path':
        k = gk.CalculateShortestPathKernel(graphs)
    elif kernel == 'vertex-gaussian':
        k = gk.CalculateVertexHistGaussKernel(graphs)
    elif kernel == 'graphlet':
        k = gk.CalculateGraphletKernel(graphs)
    elif kernel == 'connected-graphlet':
        k = gk.CalculateConnectedGraphletKernel(graphs)
    sim = np.diagonal(k, offset=len(graphs1))
    pos_sim = sim[:len(graph1)]
    neg_sim = sim[len(graph1):]
    score = sum(pos_sim>neg_sim)/len(graphs1)
    auc = compute_auc(sim, [1 for i in range(len(pos_sim))] + [0 for i in range(len(pos_sim))])
    print("%s Test Accuracy: %.5f, Test AUC: %.5f" % (kernel, score, auc))
    
size = 1000
node = 10
    
# ['IMDB-MULTI', 'COIL-DEL', 'PTC_MR', 'IMDB-BINARY']
adj, diff, feat, labels, num_nodes = load('COIL-DEL')
sample = np.random.choice(np.arange(adj.shape[0]), size=size, replace=False)
sample_n = 0
# sample_node = np.random.randint(0, adj.shape[1]-node)
adj = adj[sample, sample_n: sample_n+node, sample_n: sample_n+node]
feat = feat[sample, sample_n: sample_n+node]
diff = []
for i in tqdm(range(adj.shape[0])):
    dif = compute_pp(np.array(np.sign(adj[i])))
    diff.append(dif)
diff = np.array(diff)
adj_pos, adj_neg, diff_pos, diff_neg, graph, graph1, graph2 = generate_graph(adj, feat)

# ###################
# do kernel methods
graphs = transform(graph)
graphs1 = transform(graph1)
graphs2 = transform(graph2)
# ['WL', 'vertex-hist', 'vertex-edge-hist', 'vertex-edge-hist', 'vertex-gaussian', 'random-walk', 'e-random-walk', 'short-path', 'edge-hist', 'graphlet', 'connected-graphlet']
for kernel in ['WL', 'vertex-edge-hist']:
    cal_kernel(graphs, graphs1, graphs2, kernel=kernel)
###################

ft_size = feat[0].shape[1]
hid_units = 2048

adj = torch.FloatTensor(adj).cuda()
diff = torch.FloatTensor(diff).cuda()
feat = torch.FloatTensor(feat).cuda()

adj_pos = torch.FloatTensor(adj_pos).cuda()
diff_pos = torch.FloatTensor(diff_pos).cuda()

adj_neg = torch.FloatTensor(adj_neg).cuda()
diff_neg = torch.FloatTensor(diff_neg).cuda()

def calmodel(model, name):
    model = model.cuda()
    model.eval()
    embeds = model.embed(feat, adj, diff).detach()
    pos_embeds = model.embed(feat, adj_pos, diff_pos).detach()
    neg_embeds = model.embed(feat, adj_neg, diff_neg).detach()
    embeds = F.normalize(embeds, dim=-1)
    pos_embeds = F.normalize(pos_embeds, dim=-1)
    neg_embeds = F.normalize(neg_embeds, dim=-1)
    pos = (embeds * pos_embeds).sum(dim=-1).cpu().data.numpy()
    neg = (embeds * neg_embeds).sum(dim=-1).cpu().data.numpy()
    corr = [1 if i==True else 0 for i in pos>neg]
    x = sum(corr)
    y = len(corr)
    pos = list(pos)
    neg = list(neg)
    auc = compute_auc(neg+pos, [0 for i in range(len(neg))] + [1 for i in range(len(pos))])
    print("%s Model Accuracy: %.5f, AUC score: %.5f" % (name, sum(corr)/len(corr), auc))
    

model = Model(ft_size, hid_units)
model.load_state_dict(torch.load('./COIL-DEL-0-Ours.pkl', map_location='cuda:0'))
calmodel(model, 'OURS')

# from multi import Model as Multi
# model = Multi(ft_size, 512, 2)
# model.load_state_dict(torch.load('./IMDB-BINARY-MULTI.pkl', map_location='cuda:0'))
# calmodel(model, 'MULTI')