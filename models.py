import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F
import math
from torch import nn



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)

        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class MLP(nn.Module):
    def __init__(self, nfeats, nhids, nclasses, dropout):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(nfeats, nhids)
        self.lin2 = nn.Linear(nhids, nclasses)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.lin2(x)
        return x

class LRNet(nn.Module):
    def __init__(self, nfeat, nhid, nclasses, k, nlayers, dropout, alpha, beta, gamma, eps, lamda, num_nodes, r, init_v='mlp'):
        super(LRNet, self).__init__()
        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nclasses)
        self.vs = nn.ModuleList()
        for i in range(1):
            if init_v == 'gcn':
                self.vs.append(gnn.GCNConv(nhid, k, add_self_loops=True, cached=True))
            else:
                self.vs.append(nn.Linear(nhid, k))

        self.att = nn.Linear(nclasses, 1, bias=False)
        self.left = nn.Linear(nclasses, nclasses, bias=False)
        self.lin_a = nn.Linear(num_nodes, nhid)
        self.gamma = gamma
        self.hidden = nhid
        self.nlayers = nlayers
        self.eps = eps
        self.k = k
        self.init_v = init_v
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.drp = dropout
        self.r = r
        self.confidence = nn.Parameter(torch.Tensor([0.,0.,0.]))
        self.matrix_drp = 1 - r
        self.b = nn.Parameter(torch.Tensor([0.]))


    def forward(self, x, adj, edge_index, edge_weight, connect, confidence):
        row, col = edge_index
        x_a = self.lin_a(connect)
        x = F.dropout(x, p=self.drp, training=self.training)
        x = self.lin1(x)
        x = (1 - self.lamda) * x + self.lamda * x_a
        x = F.relu(x)
        x = F.dropout(x, p=self.drp, training=self.training)
        raw = x
        x = self.lin2(x)
        init = x

        confi_att = F.relu(self.left(x)[row] + self.left(x)[col])
        confi_att = F.sigmoid(self.att(confi_att).squeeze(-1))
        w = F.softmax(self.confidence)
        confidence = confi_att * w[0] + confidence[0] * w[1] + confidence[1] * w[2]

        for i in range(1):
            if self.init_v == "gcn":
                V = self.vs[i](raw, edge_index)
            else:
                V = self.vs[i](raw)
            U = (torch.sparse.mm(adj, V))
            inver = torch.inverse(torch.mm(V.T, V))
            U = torch.mm(U, inver)

            if self.training:
                drp = torch.bernoulli(torch.Tensor([self.matrix_drp] * x.size(1))).to(x.device) / self.matrix_drp
                drp_x = x * drp.view(1, -1)
                drp = torch.bernoulli(torch.Tensor([self.matrix_drp] * self.k)).to(x.device) / self.matrix_drp
            else:
                drp_x = x
                drp = 1

            for j in range(self.gamma):
                if j%2==0:
                    U = self.update_u(U, V, x, edge_weight, edge_index, drp_x, drp, confidence)
                else:
                    V = self.update_v(U, V, x, edge_weight, edge_index, drp_x, drp, confidence)
            xt_v = torch.mm(x.T, V)
            prop_x = torch.mm(U, xt_v.T)
            x = (1-self.alpha)*prop_x + self.alpha*init
        return x

    def update_u(self, U, V, x, edge_weight, edge_index, drp_x, drp, confidence):
        row, col = edge_index
        if self.training:
            r_v = drp.view(1, -1) * V
        else:
            r_v = V

        score = (edge_weight-((U[row] * V[col]).sum(dim=1)))
        S = score**2
        sorted_s = torch.sort(S)[0]
        edge_size = edge_weight.size(0)
        first_quartile = sorted_s[edge_size//4]
        third_quartile = sorted_s[3*edge_size//4]
        threshold = (third_quartile+1.5*(third_quartile-first_quartile))
        threshold = threshold*confidence
        confidence = torch.where(S<threshold, torch.Tensor([1.]).to(x.device), torch.Tensor([0.]).to(x.device))
        sparse_adj = torch.sparse_coo_tensor(edge_index, (score+self.b)*confidence, [x.size(0), x.size(0)])
        temp = torch.sparse.mm(sparse_adj, V) + torch.mm(U, torch.mm(V.T, V))
        U = (temp + self.eps*torch.mm(x, torch.mm(drp_x.T, V)))

        inver = torch.inverse(
            (torch.mm(V.T, V)+self.eps*torch.mm(r_v.T, V)
            + self.beta*torch.eye(self.k, device=x.device)
             )
                             )
        U = torch.mm(U, inver)
        return U

    def update_v(self, U, V, x, edge_weight, edge_index, drp_x, drp, confidence):
        if self.training:
            r_u = drp.view(1, -1) * U
        else:
            r_u = U
        inver = torch.inverse(torch.mm(U.T, U) + self.eps*torch.mm(r_u.T, U) + self.beta*torch.eye(U.size(1), device=x.device))
        row, col = edge_index
        score = (edge_weight - (V[row] * U[col]).sum(dim=1))
        S = score ** 2
        sorted_s = torch.sort(S)[0]
        edge_size = edge_weight.size(0)
        first_quartile = sorted_s[edge_size // 4]
        third_quartile = sorted_s[3 * edge_size // 4]
        threshold = (third_quartile + 1.5 * (third_quartile - first_quartile))
        threshold = threshold * confidence
        confidence = torch.where(S < threshold, torch.Tensor([1.]).to(x.device), torch.Tensor([0.]).to(x.device))
        sparse_adj = torch.sparse_coo_tensor(edge_index, (score+self.b)*confidence, [x.size(0), x.size(0)])
        temp = torch.sparse.mm(torch.transpose(sparse_adj, 0, 1), U) + \
               torch.mm(V, torch.mm(U.T, U))
        V = temp + self.eps*torch.mm(x,  torch.mm(drp_x.T, U))
        return torch.mm(V, inver)



