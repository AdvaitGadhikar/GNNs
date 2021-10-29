import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import GCN_layer
from torch.nn.utils import spectral_norm

class GCN_model(nn.Module):
    def __init__(self, input_features, num_classes, num_hidden):
        super(GCN_model, self).__init__()

        self.layer1 = GCN_layer(input_features, num_hidden)
        self.layer2 = GCN_layer(num_hidden, num_classes)

    def forward(self, x, A):
        x = F.relu(self.layer1(x, A))
        x = self.layer2(x, A)

        return x


class GCN_nifty(nn.Module):

    def __init__(self, input_features, num_classes, num_hidden):
        super(GCN_nifty, self).__init__()

        self.encoder = spectral_norm(GCN_layer(input_features, num_hidden))
        self.linear1 = spectral_norm(nn.Linear(num_hidden, num_hidden))
        self.linear2 = spectral_norm(nn.Linear(num_hidden, num_classes))

    def forward(self, x, A):
        x = self.encoder(x, A)

        return x

    def predictor(self, z):
        z = self.linear1(z)
        return z


    def classifier(self, z):

        out = self.linear2(z)
        return out