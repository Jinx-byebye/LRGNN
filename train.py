import torch
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from models import LRNet, GCN, MLP
import time
import argparse
from torch import nn
import numpy as np
import uuid
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_dense_adj, dense_to_sparse, degree
import math


checkpt_file = 'trained_model_dict/' + uuid.uuid4().hex + '.pt'
mlp_checkpt_file = 'trained_model_dict/' + uuid.uuid4().hex + 'mlp.pt'
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--patience', type=int, default=200,
                    help='Early stopping.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden unit.')
parser.add_argument('--nlayers', type=int, default=1,
                    help='Number of convolutional layers.')
parser.add_argument('--data', type=str, default='texas',
                    help='Data set to be used.')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='Strength of initial residual.')
parser.add_argument('--gamma', type=int, default=1,
                    help='Number of update iterations.')
parser.add_argument('--k', type=int, default=10,
                    help='Operating rank.')
parser.add_argument('--delta', type=float, default=0.9,
                    help='Ratio of negative edges.')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Weight of regularization term.')
parser.add_argument('--eps', type=float, default=0.1,
                    help='Weight of propagation term.')
parser.add_argument('--estimator', type=str, default='mlp',
                    help='Estimator to generate pseudo labels.')
parser.add_argument('--large_scale', type=int, default=0,
                    help='If set to 1, the signed adjacency matrix will be row-normalized.')
parser.add_argument('--lamda', type=float, default=0.5,
                    help='Strength of adj features.')
parser.add_argument('--r', type=float, default=0.5,
                    help='Dropout rate for matrix factorization.')


args = parser.parse_args()
device = 'cuda:0'

citation = ['texas', 'wisconsin', 'cornell']
if args.data in citation:
    dataset = WebKB(root='data/', name=args.data)
elif args.data in ['squirrel', 'chameleon']:
    dataset = WikipediaNetwork(root='data/', name=args.data)
elif args.data == 'actor':
    dataset = Actor(root='data/Actor')
else:
    dataset = Planetoid(root='data', name=args.data, split='geom-gcn')

data = dataset[0].to(device)
y = torch.nn.functional.one_hot(data.y).float().to(device)


def train_step(train_mask, model, optimizer, criterion, adj=None):
    model.train()
    optimizer.zero_grad()
    if adj is None:
        if args.estimator == 'mlp':
            out = model(data.x)
        else:
            out = model(data.x, sparse_adj)
    else:
        out = model(data.x, adj, edge_index, edge_weight, connect, confidence)
    loss = criterion(out[train_mask], data.y[train_mask])
    if adj is None:
        loss.backward()
    else:
        loss.backward(retain_graph=True)
    optimizer.step()
    return loss


def val_step(val_mask, model, criterion,adj=None):
    model.eval()
    with torch.no_grad():
        if adj is None:
            if args.estimator == 'mlp':
                out = model(data.x)
            else:
                out = model(data.x, sparse_adj)

        else:
            out = model(data.x, adj, edge_index, edge_weight, connect, confidence)
        pred = out.argmax(dim=1)
        loss = criterion(out[val_mask], data.y[val_mask])
        acc = int((pred[val_mask] == data.y[val_mask]).sum()) / int(val_mask.sum())
        return loss.item(), acc


def test_step(test_mask, model, criterion, adj=None):
    model.eval()
    with torch.no_grad():
        if adj is None:
            if args.estimator == 'mlp':
                out = model(data.x)
            else:
                out = model(data.x, sparse_adj)
        else:
            out = model(data.x, adj, edge_index, edge_weight, connect, confidence)
        pred = out.argmax(dim=1)
        loss = criterion(out[test_mask], data.y[test_mask])
        acc = int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum())
        return loss.item(), acc

acc = []
mlp_acc = []
data.edge_index = add_remaining_self_loops(data.edge_index)[0]


if args.estimator == 'gcn':
    dense_adj = to_dense_adj(data.edge_index)[0]
    deg = dense_adj.sum(dim=1, keepdim=True)
    deg = deg.pow(-0.5)
    dense_adj = deg.view(-1, 1) * dense_adj * deg.view(1, -1)
    sparse_adj = dense_adj.to_sparse()
# row, col = data.edge_index
# deg = degree(col, num_nodes=y.size(0))
# deg_inv_sqrt = deg.pow(-0.5)
# deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
# norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
# sparse_adj = torch.sparse_coo_tensor(data.edge_index, norm, size=[y.size(0), y.size(0)])

def train_mlp():
    # print(f'training {args.estimator}...')
    best = 100
    patience = args.patience
    count = 0
    for i in range(1000):
        _ = train_step(train_mask, predictor, mlp_optimizer, criterion=criterion_mlp)
        val_loss, val_acc = val_step(val_mask, predictor, criterion=criterion_mlp)
        if val_loss < best:
            count = 0
            best = val_loss
            torch.save(predictor.state_dict(), mlp_checkpt_file)
        else:
            count += 1
            if count == patience:
                break

    predictor.load_state_dict(torch.load(mlp_checkpt_file))
    loss, best_acc = test_step(test_mask, predictor, criterion=criterion_mlp)
    mlp_acc.append(best_acc)
    # print(f'{args.estimator} acc: {best_acc:.3f}')


edge_index, _ = remove_self_loops(data.edge_index)
row, col = edge_index
connect = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index[0]),
                                      size=[y.size(0), y.size(0)]).float()


for i in range(10):
    split = i%10
    train_mask = data.train_mask[:, split]
    val_mask = data.val_mask[:, split]
    test_mask = data.test_mask[:, split]
    criterion = nn.CrossEntropyLoss()
    criterion_mlp = nn.CrossEntropyLoss()
    if args.estimator == 'mlp':
        predictor = MLP(data.num_features, 64, dataset.num_classes, args.dropout).to(device)
    else:
        predictor = GCN(data.num_features, 64, dataset.num_classes, args.dropout).to(device)
    if args.data in ['cora', 'citeseer', 'pubmed']:
        init_v = 'gcn'
    else:
        init_v ='mlp'
    model = LRNet(data.num_features, args.hidden, dataset.num_classes,
                   dropout=args.dropout, k=args.k, nlayers=args.nlayers,
                   gamma=args.gamma, alpha=args.alpha, beta=args.beta,
                   eps=args.eps, lamda=args.lamda, num_nodes=data.num_nodes, r=args.r, init_v=init_v).to(device)
    mlp_optimizer = torch.optim.Adam(params=predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gnn_optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_mlp()
    if args.estimator == 'mlp':
        mlp_out = predictor(data.x)
    else:
        mlp_out = predictor(data.x, sparse_adj)
    pseudo_labels = mlp_out.softmax(dim=1)

    mask = torch.where(train_mask == True, torch.Tensor([1.]).to(device),
                       torch.Tensor([0.]).to(device)).unsqueeze(-1)
    pseudo_labels = mask * y + (1 - mask) * pseudo_labels
    edge_weight = (pseudo_labels[row] * pseudo_labels[col]).sum(dim=1)
    edge_weight = edge_weight - args.delta

    confidence = torch.linalg.vector_norm(pseudo_labels, dim=1) - 1 / dataset.num_classes
    confidence = confidence[row] * confidence[col]
    confidence = torch.stack([confidence, torch.log(math.e - 1 + torch.abs(edge_weight))], dim=0)

    if args.large_scale == 1:
        deg = degree(col, num_nodes=data.num_nodes)
        edge_weight = edge_weight/deg[col]
        edge_weight[edge_weight<-5] = -5
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=[y.size(0), y.size(0)])

    best = 100
    best_acc = 0
    patience = args.patience
    best_val_acc = 0
    count = 0
    total_time = time.time()
    times = []
    for j in range(1000):
      t = time.time()
      train_loss = train_step(train_mask, model, gnn_optimizer, criterion, adj=adj)
      times.append(time.time()-t)
      val_loss, val_acc = val_step(val_mask, model, criterion, adj=adj)
      if val_loss<best:
          count = 0
          best = val_loss
          best_val_acc = val_acc
          torch.save(model.state_dict(), checkpt_file)

      else:
          count += 1
          if count==patience:
              break
    # print(np.mean(times*1000))
    # print(time.time()-total_time)

    model.load_state_dict(torch.load(checkpt_file))
    loss, best_acc = test_step(test_mask, model, criterion,adj=adj)
    acc.append(best_acc)
    print('---------------------------------------')
    print(f'intermediate result: {best_acc}')

print('---------------------------------------')
print(f'{args.data}: mean acc:{np.mean(acc)}')
print(np.std(np.array(acc)))
