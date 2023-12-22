# -*- coding: utf-8 -*-

# ----------------------------
# Import necessary libraries
# ----------------------------
import torch
import torch.nn as nn


# ----------------------------
# 2-1 Preparation
# ----------------------------
print("=======2-1 Preparation=======")

class SimpleMlp(nn.Module):
    def __init__(self, vec_length:int=16, hidden_unit_1:int=8, hidden_unit_2:int=2): 
        """
        Arguments:
            vec_length: Length of the input vector
            hidden_unit_1: Number of neurons in the first linear layer
            hidden_unit_2: Number of neurons in the second linear layer
        """
        # Call to the __init__() method of the inherited nn.Module
        super(SimpleMlp, self).__init__()
        # First linear layer
        self.layer1 = nn.Linear(vec_length, hidden_unit_1)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Second linear layer
        self.layer2 = nn.Linear(hidden_unit_1, hidden_unit_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation follows the sequence: linear layer → ReLU → linear layer
        Arguments:
            x: Input. Shape: (B, D_in)
                B: Batch size, D_in: Length of the vector
        Returns:
            out: Output. Shape: (B, D_out)
                B: Batch size, D_out: Length of the vector
        """
        # First linear layer
        out = self.layer1(x)
        # ReLU
        out = self.relu(out)
        # Second linear layer
        out = self.layer2(out) 
        return out

vec_length = 16  # Length of the input vector
hidden_unit_1 = 8  # Number of neurons in the first linear layer
hidden_unit_2 = 2  # Number of neurons in the second linear layer

batch_size = 4  # Batch size. Number of input vectors

# Input vector. Shape of x: (4, 16)
x = torch.randn(batch_size, vec_length)
# Define the MLP
net = SimpleMlp(vec_length, hidden_unit_1, hidden_unit_2) 
# Forward propagation in the MLP
out = net(x)
# Check that the shape of the MLP output out is (4, 2)
print(out.shape)