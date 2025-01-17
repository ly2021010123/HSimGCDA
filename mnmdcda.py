import torch as th
from torch import nn
from dgl import function as fn

class ConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, k=2, method='sum', bias=True, batchnorm=False, activation='relu',
                 dropout=0.0):
        super(ConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k + 1
        self.method = method
        self.weights = []
        for i in range(self.k):
            self.weights.append(nn.Parameter(th.Tensor(in_feats, out_feats)))
        self.biases = None
        self.activation = None
        self.batchnorm = None
        self.dropout = None
        if bias:
            self.biases = []
            for i in range(self.k):
                self.biases.append(nn.Parameter(th.Tensor(out_feats)))
        self.reset_parameters()

        if activation == 'relu':
            self.activation = th.relu
        if batchnorm:
            if method == 'cat':
                self.batchnorm = nn.BatchNorm1d(out_feats * self.k)
            else:
                self.batchnorm = nn.BatchNorm1d(out_feats)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for i in range(self.k):
            nn.init.xavier_uniform_(self.weights[i])
            if self.biases is not None:
                nn.init.zeros_(self.biases[i])

    def forward(self, graph, feat):
        with graph.local_scope():
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            if self.biases is not None:
                rst = th.matmul(feat, self.weights[0]) + self.biases[0]
            else:
                rst = th.matmul(feat, self.weights[0])
            for i in range(1, self.k):
                feat = feat * norm
                graph.ndata['h'] = feat
                if 'e' in graph.edata.keys():
                    graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm
                if self.method == 'sum':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = rst + y
                elif self.method == 'mean':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = rst + y
                    rst = rst / self.k
                elif self.method == 'cat':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = th.cat((rst, y), dim=1)
                    rst_emb_feas = rst.data
                    rst_emb_feas = rst_emb_feas.numpy()

            if self.batchnorm is not None:
                rst = self.batchnorm(rst)
            if self.activation is not None:
                rst = self.activation(rst)
            if self.dropout is not None:
                rst = self.dropout(rst)
            return rst

class GraphEmbbeding(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, k, method, bias, batchnorm, activation, num_layers, dropout):
        super(GraphEmbbeding, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                hid_feats = out_feats
            self.layers.append(ConvLayer(in_feats, hid_feats, k, method, bias, batchnorm, activation, dropout))
            if method == 'cat':
                in_feats = hid_feats * (k + 1)
            else:
                in_feats = hid_feats

    def forward(self, graph, feat):
        for i, layer in enumerate(self.layers):
            feat = layer(graph, feat)
        return feat

class MNMDCDA(nn.Module):
    def __init__(self, args):
        super(MNMDCDA, self).__init__()
        self.args = args
        self.lin_c=nn.Linear(args.circRNA_number,args.in_feats,bias=False)
        self.lin_d=nn.Linear(args.disease_number,args.in_feats,bias=False)
        self.gcn_cc = GraphEmbbeding(args.circRNA_number, args.hid_feats, args.out_feats, args.k, args.method, args.gcn_bias,
                                     args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)
        self.gcn_dd = GraphEmbbeding(args.disease_number, args.hid_feats, args.out_feats, args.k, args.method, args.gcn_bias,
                                     args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)
        self.gcn_cd = GraphEmbbeding(args.in_feats, args.hid_feats, args.out_feats, args.k, args.method, args.gcn_bias,
                                     args.gcn_batchnorm, args.gcn_activation, args.num_layers, args.dropout)

        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat = 4 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            self.mlp.add_module(str(idx), nn.Linear(args.out_feats * (args.k+1) *  4, out_feat))
            in_feat = out_feat
        self.mlp.add_module('sigmoid', nn.Sigmoid())
    def forward(self, cc_graph, dd_graph, cd_graph, circRNA, disease, tran_sample):
        emb_cc_sim = self.gcn_cc(cc_graph, circRNA)
        emb_cc_sim_feas = emb_cc_sim.data
        emb_cc_sim_feas = emb_cc_sim_feas.numpy()

        emb_dd_sim = self.gcn_dd(dd_graph, disease)
        emb_dd_sim_feas = emb_dd_sim.data
        emb_dd_sim_feas = emb_dd_sim_feas.numpy()

        emb_ass = self.gcn_cd(cd_graph, th.cat((self.lin_c(circRNA),self.lin_d(disease)), dim=0))
        emb_ass_sim_feas = emb_ass.data
        emb_ass_sim_feas = emb_ass_sim_feas.numpy()

        emb_cc_ass = emb_ass[:self.args.circRNA_number, :]
        emb_cc_ass_sim_feas = emb_cc_ass.data
        emb_cc_ass_sim_feas = emb_cc_ass_sim_feas.numpy()
        emb_dd_ass = emb_ass[self.args.circRNA_number:, :]
        emb_dd_ass_sim_feas = emb_dd_ass.data
        emb_dd_ass_sim_feas = emb_dd_ass_sim_feas.numpy()

        emb_cc = th.cat((emb_cc_sim, emb_cc_ass), dim=1)
        emb_dd = th.cat((emb_dd_sim, emb_dd_ass), dim=1)
        train_sample_emb = th.cat((emb_cc[tran_sample[:, 0]], emb_dd[tran_sample[:, 1]]), dim=1)
        train_sample_emb_feas = train_sample_emb.data
        train_sample_emb_feas = train_sample_emb_feas.numpy()

        train_sample_train_score = self.mlp(train_sample_emb)
        train_sample_train_score_feas = train_sample_train_score.data
        train_sample_train_score_feas = train_sample_train_score_feas.numpy()

        train_sample_predicted_labels = (train_sample_train_score_feas > 0.5).astype(int)

        return train_sample_train_score, train_sample_train_score_feas, train_sample_predicted_labels, train_sample_emb