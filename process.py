import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adjCN(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(adj).tocoo()

def normalize_adjSalton (adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    Dtemp = d_mat_inv_sqrt @ d_mat_inv_sqrt.T

    CNmat = adj @ adj
    result = CNmat @ Dtemp
    return result

def mymaximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data <= 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def myminimum(A,B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data >= 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def normalize_adjHDI(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    deg_row = np.tile(rowsum, (1,adj.shape[0]))
    deg_row = sp.coo_matrix(deg_row)
    sim = adj.dot(adj)
    X = sim.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    deg_row = mymaximum(deg_row, deg_row.T)

    sim = sim/deg_row
    whereAreNan = np.isnan(sim)
    whereAreInf = np.isinf(sim)
    sim[whereAreNan] = 0
    sim[whereAreInf] = 0
    
    sim = sp.coo_matrix(sim)
    return sim

def normalize_adjHPI(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    deg_row = np.tile(rowsum, (1,adj.shape[0]))
    deg_row = sp.coo_matrix(deg_row)
    sim = adj.dot(adj)
    X = sim.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    deg_row = myminimum(deg_row, deg_row.T)
    sim = sim/deg_row
    whereAreNan = np.isnan(sim)
    whereAreInf = np.isinf(sim)
    sim[whereAreNan] = 0
    sim[whereAreInf] = 0

    sim = sp.coo_matrix(sim)
    return sim