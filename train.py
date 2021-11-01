import time 
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score

from models import *
from utils import *
from dropout_adj import drop_edges


"""
Fair Metric
"""
def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

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
model = GCN_model(input_features, num_classes, num_hidden)

adj.to(device)
features.to(device)
labels.to(device)
model.to(device)

"""
Train the GCN model
"""
num_epochs = 1000
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

    if i % 10 == 0:
        val_output = model(features, adj)
        val_loss = bce_loss(val_output[idx_val, :], labels[idx_val])
        val_predicted = torch.argmax(val_output[idx_val], dim=1)
        val_acc = (val_predicted == labels[idx_val]).sum() / idx_val.shape[0]
        print('Val Loss: %.4f, Acc: %.4f' %(val_loss, val_acc))

model.eval()
output = model(features.to(device), adj.to(device))
counter_features = features.clone()
counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
counter_output = model(counter_features.to(device), adj.to(device))
noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
noisy_output = model(noisy_features.to(device), adj.to(device))


output_preds = torch.argmax(output, dim=1).type_as(labels)
counter_output_preds = torch.argmax(counter_output, dim=1).type_as(labels)
noisy_output_preds = torch.argmax(noisy_output, dim=1).type_as(labels)

auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output_preds.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())

print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Parity: {parity} | Equality: {equality}')
print(f'F1-score: {f1_s}')
print(f'CounterFactual Fairness: {counterfactual_fairness}')
print(f'Robustness Score: {robustness_score}')

result_list = np.array([auc_roc_test, parity, equality, f1_s, counterfactual_fairness, robustness_score])
np.save('results_list_gnn.npy', result_list, allow_pickle=True)


