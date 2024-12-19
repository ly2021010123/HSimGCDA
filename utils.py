import random
import numpy as np
import torch as th

def data_processing(data, args):
    cd_matrix = data['cd_matrix']
    one_index = []
    zero_index = []
    print(cd_matrix.shape[0])
    for i in range(cd_matrix.shape[0]):
        for j in range(cd_matrix.shape[1]):
            if cd_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    print(len(one_index))
    print(one_index)
    random.shuffle(zero_index)
    print(len(zero_index))
    print(zero_index)
    unsamples=[]
    if args.negative_rate == -1:
        zero_index = zero_index
    else:
        unsamples = zero_index[int(args.negative_rate * len(one_index)):]
        print(len(unsamples))
        zero_index = zero_index[:int(args.negative_rate * len(one_index))]
        print(len(zero_index))
    index = np.array(one_index + zero_index, np.int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=np.int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    cd = samples[samples[:, 2] == 1, :2]
    print(len((cd_matrix.transpose())))
    print(cd_matrix.transpose())

    circRNAs = data['circRNAs']
    diseases = data['diseases']
    data['circRNAs'] = circRNAs
    data['diseases'] = diseases
    data['train_cd'] = cd
    data['unsamples']=np.array(unsamples)

    np.random.shuffle(samples)
    data['train_samples'] = samples

def k_matrix(matrix, k=20):
    num = matrix.shape[0]

    print(matrix.shape)
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def get_data(args):
    data = dict()
    cf = np.loadtxt(args.data_dir + 'circRNA functional similarity matrix.txt', dtype=np.float)
    dtype = {'names': ('number', 'string'), 'formats': (np.float, 'U100')}
    c_number = np.genfromtxt(args.data_dir + 'circRNA number.txt', dtype=dtype, delimiter='\t')

    c_name = c_number['number']
    c_name_int = c_name.astype(int)
    d_number = np.genfromtxt(args.data_dir + 'disease number.txt', dtype=dtype, delimiter='\t')
    d_name = d_number['number']
    d_name_int = d_name.astype(int)

    dss = np.loadtxt(args.data_dir + 'disease semantic similarity matrix.txt', dtype=np.float)
    cd_matrix = np.loadtxt(args.data_dir + 'cd_matrix.txt', dtype=np.float)
    circRNAs = np.loadtxt(args.data_dir + 'circRNAs.txt', dtype=np.float)
    diseases = np.loadtxt(args.data_dir + 'diseases.txt', dtype=np.float)
    data['circRNA_number'] = int(cf.shape[0])
    data['disease_number'] = int(dss.shape[0])

    data['c_name_int'] =  c_name_int
    data['d_name_int'] =  d_name_int
    data['cf'] = cf
    data['dss'] = dss
    data['cd_matrix'] = cd_matrix
    data['circRNAs'] = circRNAs
    data['diseases'] = diseases

    return data