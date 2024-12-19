import argparse
import numpy as np
import torch.nn.functional as F
from mnmdcda import MNMDCDA
from torch import optim,nn
from utils import k_matrix
from utils import get_data,data_processing
from sklearn.metrics import auc
from embed_train import ConVol_Matrix
from embed_train import  _embedding_training
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy import interp

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch as t
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import KFold
import warnings
import dgl
import networkx as nx
import copy
import numpy as np
import torch
import torch as th
import scipy.io as scio

warnings.filterwarnings("ignore")
device = t.device('cuda:0' if t.cuda.is_available() else "cpu")
t.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--wd', type=float, default=0.001, help='weight_decay')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument("--in_feats", type=int, default=64, help='Input layer dimensionalities.')
parser.add_argument("--hid_feats", type=int, default=64, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=64, help='Output layer dimensionalities.')
parser.add_argument("--method", default='cat', help='Merge feature method')
parser.add_argument("--gcn_bias", type=bool, default=True, help='gcn bias')
parser.add_argument("--gcn_batchnorm", type=bool, default=True, help='gcn batchnorm')
parser.add_argument("--gcn_activation", default='relu', help='gcn activation')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[1280, 1], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=20, help='neighbor')

#############################################
parser.add_argument('--SAGCNmethod',  default='SAGCN', help='The embedding learning method')
parser.add_argument('--task', default='circRNA-disease-prediction',
                    help='Choose to evaluate the embedding quality based on a specific prediction task.')
parser.add_argument('--seed', default=0, type=int, help='seed value')
parser.add_argument('--SAGCN_embed_tech',  default='CN', help='The embedding techniques of SAGCN')
#############################################
parser.add_argument('--MLP_epochs', type=int, default=65, metavar='N', help='number of epochs to train')
parser.add_argument("--MLP_hid_feats", type=int, default=64, help='Hidden layer dimensionalities.')
parser.add_argument('--dataset', default='1-CircR2Disease_All Species_MeSH', help='dataset')
parser.add_argument('--negative_rate', type=float,default=1.0, help='negative_rate')

args = parser.parse_args()
args.dd2=True
args.data_dir = 'data/' + args.dataset + '/'

def loading(args):
    data = get_data(args)
    print(data)
    args.circRNA_number = data['circRNA_number']
    args.disease_number = data['disease_number']
    data_processing(data,args)
    return data

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    dataset = loading(args)

    model = MNMDCDA(args).to(device)
    optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd,
                            lr=args.lr)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    cross_entropy = nn.BCELoss(reduction='mean')
    data = dataset
    circRNA = data['circRNAs']
    disease = data['diseases']
    cc_matrix = k_matrix(data['circRNAs'], args.neighbor)
    cc_ConVol_Matrix = ConVol_Matrix(args, cc_matrix)
    dd_matrix = k_matrix(data['diseases'], args.neighbor)
    dd_ConVol_Matrix = ConVol_Matrix(args, dd_matrix)
    cc_nx = nx.from_numpy_matrix(cc_ConVol_Matrix)
    dd_nx = nx.from_numpy_matrix(dd_ConVol_Matrix)
    cc_graph = dgl.from_networkx(cc_nx)
    dd_graph = dgl.from_networkx(dd_nx)

    cd_copy = copy.deepcopy(data['train_cd'])
    cd_copy[:, 1] = cd_copy[:,1] + args.circRNA_number
    u = np.concatenate((cd_copy[:, 0], cd_copy[:, 1]))
    v = np.concatenate((cd_copy[:, 1], cd_copy[:, 0]))
    test_cd_graph = dgl.graph((u, v), num_nodes=args.circRNA_number + args.disease_number)
    adj_matrix = test_cd_graph.adjacency_matrix().to_dense().numpy()
    cd_ConVol_Matrix = ConVol_Matrix(args, adj_matrix)
    cd_nx = nx.from_numpy_matrix(cd_ConVol_Matrix)
    cd_graph = dgl.from_networkx(cd_nx)
    circRNA_th = th.Tensor(circRNA)
    disease_th = th.Tensor(disease)

    AUC = 0
    auprc = 0
    acc = 0
    f1 = 0
    recall = 0
    pre = 0
    max_train_acc = 0

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    All_train_features = []
    All_test_features = []
    All_train_labels = []
    All_test_labels = []
    pred_labels = []
    probas = []
    AllACC = []
    AllAUC = []
    AllAUPR = []
    AllMCC = []
    AllSN = []
    AllSP = []
    AllPE = []
    AllFPR = []
    AllF1_score = []
    AllResults = {}

    train_sample = dataset['train_samples'][:, :2]
    train_label = dataset['train_samples'][:, 2]
    train_label_reshaped = np.expand_dims(train_label, axis=1)
    train_samples = np.hstack((train_sample, train_label_reshaped))

    for i in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_score, train_score_feas, Train_predicted_label, train_embeddings = model(cc_graph, dd_graph, cd_graph, circRNA_th, disease_th, train_samples)
        train_score = train_score.squeeze(1)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        train_label = t.tensor(train_label).to(device)
        train_label = train_label.float()

        train_loss = cross_entropy(train_score, train_label)
        train_loss.backward()
        train_acc = accuracy_score(train_label, Train_predicted_label)
        train_auc = roc_auc_score(train_label, train_score_feas)
        optimizer.step()
        train_aupr = average_precision_score(train_label.detach().cpu().numpy(), train_score.detach().cpu().numpy())
        train_f1 = f1_score(train_label.detach().cpu().numpy(),
                           np.rint(train_score.detach().cpu().numpy()).astype(np.int64), average='macro')
        train_recall = recall_score(train_label.detach().cpu().numpy(),
                                   np.rint(train_score.detach().cpu().numpy()).astype(np.int64), average='macro')
        train_pre = precision_score(train_label.detach().cpu().numpy(),
                                   np.rint(train_score.detach().cpu().numpy()).astype(np.int64), average='macro')
        train_mcc = matthews_corrcoef(train_label, Train_predicted_label)

        if train_acc > max_train_acc:
            t.save(model.state_dict(), "./save_model/5_fold/train_model.pth")
            max_test_acc = train_acc
            acc = train_acc
            AUC = train_auc
            auprc = train_aupr
            f1 = train_f1
            recall = train_recall
            pre = train_pre
            MCC = train_mcc
            print(f'Epoch: {i + 1:03d}/{args.epochs:03d}' f'   | Learning Rate {scheduler.get_last_lr()[0]:.6f}')
            print(f'Train Auc.: {train_auc:.4f}' f' | Train Acc.: {train_acc:.4f}')
            print(f'Train Loss.: {train_loss.item():.4f}')
        scheduler.step()

    emb = train_embeddings
    emb_features = emb.data
    emb_features = emb_features.numpy()
    temp = 2
    emb_labels =  train_samples[:, temp]
    emb_labels_reshaped = np.expand_dims(emb_labels, axis=1)
    dataset = np.hstack((emb_features, emb_labels_reshaped))
    scio.savemat('emb_features.mat', mdict={'emb_features': emb_features})
    scio.savemat('emb_labels.mat', mdict={'emb_labels': emb_labels})

    temp = 1280
    X = dataset[:, :temp]
    y = dataset[:, temp]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tran_feas = []
    tran_labels = []
    test_feas = []
    test_labels = []
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f'Fold CV: {fold}/{kf.n_splits}')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tran_feas.append(X_train)
        tran_labels.append(y_train)
        test_feas.append(X_test)
        test_labels.append(y_test)
        train_feas_tensor = t.Tensor(X_train)
        y_train_tensor = t.Tensor(y_train)
        test_feas_tensor = t.Tensor(X_test)

        input_size = temp
        hidden_size = args.MLP_hid_feats
        output_size = 1
        dropout_prob = 0.5
        model = MLPClassifier(input_size, hidden_size, output_size, dropout_prob)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(args.MLP_epochs):
            optimizer.zero_grad()
            outputs = model(train_feas_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        model.eval()
        with torch.no_grad():
            predicted_probs = model(test_feas_tensor).squeeze().numpy()
            predicted_labels = (predicted_probs > 0.5).astype(int)
            pred_labels.append(predicted_labels)
            probas.append(predicted_probs)
            test_acc = accuracy_score(y_test, predicted_labels)
            test_auc = roc_auc_score(y_test, predicted_probs)
            test_aupr = average_precision_score(y_test, predicted_probs)
            test_f1 = f1_score(y_test,
                               np.rint(predicted_probs).astype(np.int64), average='macro')
            test_recall = recall_score(y_test,
                                       np.rint(predicted_probs).astype(np.int64), average='macro')
            test_pre = precision_score(y_test,
                                       np.rint(predicted_probs).astype(np.int64), average='macro')
            test_mcc = matthews_corrcoef(y_test, predicted_labels)
            acc = test_acc
            AllACC.append(acc)
            AUC = test_auc
            AllAUC.append(AUC)
            auprc = test_aupr
            AllAUPR.append(auprc)
            f1 = test_f1
            AllF1_score.append(f1)
            recall = test_recall
            AllSN.append(recall)
            pre = test_pre
            AllPE.append(pre)
            MCC = test_mcc
            AllMCC.append(MCC)
            i = fold - 1
            fpr, tpr, thresholds = roc_curve(test_labels[i], predicted_probs)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1.0, alpha=1,
                     label='ROC %d fold  (AUC = %0.4f)' % (i + 1, roc_auc))
            print("-----------------------------------------------\n")
            print("fold =", i + 1)
            print("-----------------------------------------------\n")
    All_train_features = tran_feas
    All_test_features = test_feas
    All_train_labels = tran_labels
    All_test_labels = test_labels
    pred_labels = pred_labels
    probas = probas
AllACC_mean = np.mean(AllACC)
AllAUC_mean = np.mean(AllAUC)
AllMCC_mean = np.mean(AllMCC)
AllSN_mean = np.mean(AllSN)
AllPE_mean = np.mean(AllPE)
AllF1_score_mean = np.mean(AllF1_score)
AllACC_std = np.std(AllACC, ddof=1)
AllAUC_std = np.std(AllAUC, ddof=1)
AllMCC_std = np.std(AllMCC, ddof=1)
AllSN_std = np.std(AllSN, ddof=1)
AllPE_std = np.std(AllPE, ddof=1)
AllF1_score_std = np.std(AllF1_score, ddof=1)

AllResults = {'pred_labels': pred_labels, 'probas': probas, 'AllACC': AllACC, 'AllAUC': AllAUC, 'AllMCC': AllMCC,
          'AllSN': AllSN,
          'AllPE': AllPE,  'AllF1_score': AllF1_score, 'AllACC_mean': AllACC_mean,
          'AllAUC_mean': AllAUC_mean,
          'AllMCC_mean': AllMCC_mean, 'AllSN_mean': AllSN_mean,
          'AllPE_mean': AllPE_mean,
           'AllF1_score_mean': AllF1_score_mean, 'AllACC_std': AllACC_std,
          'AllAUC_std': AllAUC_std,
          'AllMCC_std': AllMCC_std, 'AllSN_std': AllSN_std,  'AllPE_std': AllPE_std,
          'AllF1_score_std': AllF1_score_std}
scio.savemat('All_train_features.mat', mdict={'All_train_features': All_train_features})
scio.savemat('All_test_features.mat', mdict={'All_test_features': All_test_features})
scio.savemat('All_train_labels.mat', mdict={'All_train_labels': All_train_labels})
scio.savemat('All_test_labels.mat', mdict={'All_test_labels': All_test_labels})
scio.savemat('AllResults.mat', mdict={'AllResults': AllResults})
plt.plot([0, 1], [0, 1], linestyle='--', label='Chance', lw=1.0,
         color='dimgray', alpha=0.5)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr,
         label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (AllAUC_mean, AllAUC_std),
         lw=1.5, alpha=1)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('./Figures/'+ 'CircR2Disease_ROC.png')
plt.show()
print(f' | Test Acc of 5 CV: {AllACC_mean:.4f}')
print(f' | Test Auc of 5 CV: {AllAUC_mean:.4f}')
print(f' | Test Recall of 5 CV: {AllSN_mean:.4f}')
print(f' | Test Precision of 5 CV: {AllPE_mean:.4f}')
print(f' | Test MCC of 5 CV: {AllMCC_mean:.4f}')
print(f' | Test F1 of 5 CV: {AllF1_score_mean:.4f}')