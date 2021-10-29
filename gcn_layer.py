import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class GCN_layer(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(GCN_layer, self).__init__()

        self.weight = nn.Parameter(torch.empty(input_features, output_features))
        self.b = nn.Parameter(torch.empty(output_features))
        std_dev = 1 / np.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight, -std_dev, std_dev)
        nn.init.uniform_(self.b, -std_dev, std_dev)

    def forward(self, x, A):
        """
        x: input data
        A: Adjacency matrix (normalized)
        """
        agg = torch.sparse.mm(A, x)
        output = torch.mm(x, self.weight) + self.b

        return output
        