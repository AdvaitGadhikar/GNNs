import time 
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from models import *
from utils import *

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


print(features.shape)
print(adj.shape)
print(labels.shape, labels.max(), labels.min())
print(sens.shape, sens)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Load GCN model
"""
input_features = features.shape[1]
num_classes = labels.unique().shape[0]
num_hidden = 16
model = GCN_model(input_features, num_classes, num_hidden)

print('adjacency matrix')
print(adj)
print(adj.shape)
print(idx_train.shape, idx_val.shape, idx_test.shape, idx_sens_train.shape)
adj.to(device)
features.to(device)
labels.to(device)
model.to(device)

"""
Train the GCN model
"""
num_epochs = 100
learning_rate = 1e-3
bce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(features, adj)
    epoch_loss = bce_loss(output[idx_train, :], labels[idx_train])
    predicted_labels = torch.argmax(output[idx_train, :], dim = 1)
    epoch_acc = (predicted_labels == labels[idx_train]).sum() / idx_train.shape[0]
    epoch_loss.backward()
    optimizer.step()

    print('Epoch: %.4f, Loss: %.4f, Acc: %.4f' %(i, epoch_loss, epoch_acc))


val_output = model(features, adj)
val_loss = bce_loss(val_output[idx_val, :], labels[idx_val])
val_predicted = torch.argmax(val_output[idx_val], dim=1)
val_acc = (val_predicted == labels[idx_val]).sum() / idx_val.shape[0]
print('Val Loss: %.4f, Acc: %.4f' %(val_loss, val_acc))