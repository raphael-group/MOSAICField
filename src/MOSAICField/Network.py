import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class DisplacementField(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_list, 
        activation_fn=nn.ReLU(),
    ):
        super(DisplacementField, self).__init__()

        layer_dim_list = [dim] + hidden_list + [dim]
        layers = []
        for l in range(len(layer_dim_list)-1):
            layers.append(nn.Linear(layer_dim_list[l], layer_dim_list[l+1]))
            if l < len(layer_dim_list)-2:
                layers.append(activation_fn)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    