import torch

from utils import *



def drop_edges(A, p):
    """
    A is a sparse torch matrix 
    p is the probability of dropping an edge
    """
    A = A.to_dense()
    mask = p * torch.ones_like(A)
    mask = torch.bernoulli(mask)
    A = A * (1 - mask)
    return A.to_sparse()