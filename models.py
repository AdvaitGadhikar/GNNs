import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from gcn_layer import GCN_layer

def GCN_model(nn.Module):
    def __init__(self, input_features, num_classes, num_hidden):
        super(GCN_model, self).__init__()

        self.layer1 = gcn_layer(input_features, num_hidden)
        self.layer2 = gcn_layer(num_hidden, num_classes)

    def forward(self, x, A):
        x = F.relu(self.layer1(x, A))
        x = self.layer2(x, A)

        return x

