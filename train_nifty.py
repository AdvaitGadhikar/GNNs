import time 
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from models import *
from utils import *
# from dropout_adj import *

"""
Load the Credit Dataset
"""
sens_attr = "Age"  # column number after feature process is 1
sens_idx = 1
predict_attr = 'NoDefaultNextMonth'
label_number = 6000
path_credit = "./credit"
adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_credit('credit', sens_attr,
                                                                        predict_attr, path=path_credit,
                                                                        label_number=label_number
                                                                        )
norm_features = feature_norm(features)
norm_features[:, sens_idx] = features[:, sens_idx]
features = norm_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Load GCN model
"""
input_features = features.shape[1]
num_classes = labels.unique().shape[0]
num_hidden = 16
model = GCN_nifty(input_features, num_classes, num_hidden)

adj.to(device)
features.to(device)
labels.to(device)
model.to(device)


"""
Augmented Graph
"""
def perturb_features(x, prob):
    x = x.clone()

    mask = torch.zeros(x.shape[1]).uniform_(0, 1) < prob
    delta = torch.ones(1).normal_(0, 1)
    idx = torch.where(mask == 1)
    x[:, mask] += delta 

    return x

def flip_sens(x, sens_idx):
    x = x.clone()
    x[:, sens_idx] = 1 - x[:, sens_idx]

    return x


"""
Train the GCN Nifty model
"""
drop_edge_rate_1 = 0.001
drop_feature_rate_1 = 0.1
sim_coeff = 0.6

num_epochs = 100
learning_rate = 1e-3
bce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(num_epochs):
    optimizer.zero_grad()
    # adj_perturbed = dropout_adj(adj, drop_edge_rate_1)
    x1 = flip_sens(perturb_features(features, drop_feature_rate_1), sens_idx)

    z = model(features, adj)
    z1 = model(x1, adj)

    tz = model.predictor(z)
    tz1 = model.predictor(z1)
    Ls = - 0.5*(F.cosine_similarity(tz, z1.detach(), dim=-1).mean() + F.cosine_similarity(tz1, z.detach(), dim=-1).mean())

    output = model.classifier(z)
    output1 = model.classifier(z1)

    closs = bce_loss(output[idx_train, :], labels[idx_train])
    closs1 = bce_loss(output1[idx_train, :], labels[idx_train])
    Lc = 0.5*closs + 0.5*closs1

    loss = (1-sim_coeff)*Ls + sim_coeff*Lc
    loss.backward()
    optimizer.step()

    print('Epoch: %.4f, Loss: %.4f, Fairness Loss: %.4f, Classificn Loss: %.4f' %(i, loss, Ls, Lc))