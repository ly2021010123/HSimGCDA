# -*- coding: utf-8 -*-
import process
import time
import scipy.sparse as sp
import torch
import networkx as nx

def load_graph(args, SimMatrix, weighted=False):
    print('#' * 70)
    print('Embedding Method: %s, Evaluation Task: %s' % (args.method, args.task))
    print('#' * 70)

    SimMatrix_nx = nx.from_numpy_matrix(SimMatrix)
    G = SimMatrix_nx
    node_num, edge_num = len(G.nodes), len(G.edges)

    print('Original Graph: nodes:', node_num, 'edges:', edge_num)
    print("Loading training graph for learning embedding...")

    node_list = list(G.nodes)
    print(len(node_list))

    adj = nx.adjacency_matrix(G, nodelist=node_list)

    g = (adj, node_list)
    print("Graph Loaded...")
    G = g
    return G

def ConVol_Matrix(args, SimMatrix):
    Graph = load_graph(args, SimMatrix)
    techniques = [args.SAGCN_embed_tech]
    for x in techniques:
        print(f'embedding techniques: {x}' )

        args.embTech = x
        SA_Matrix = _embedding_training(args, SimMatrix, Graph)

        print('获得了: %s 的相似性评估的卷积矩阵' % args.embTech)
        return SA_Matrix

def _embedding_training(args, SimMatrix, G_=None):
    seed=args.seed
    if args.SAGCNmethod == 'SAGCN':

        sparse = True
        adj = G_[0]

        if args.embTech == 'OGCN':
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'CN':
            adj = process.normalize_adjCN(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'HDS':
            adj = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'HPS':
            adj = process.normalize_adjHPI(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Salton':
            adj = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
        else:
            print("No such embedding technique \n We are calling default DGI", args.embTech)
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        if sparse:
            adj_array = adj.toarray()
        else:
            adj_array = adj
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])

    return adj_array
