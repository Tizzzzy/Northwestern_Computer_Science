import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class FFNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(FFNN, self).__init__()
        # Define input layer:
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        
    def sigmoid(self, z):
        #z = z.detach().numpy()
        return 1.0/(1.0 + torch.exp(-z))
    
    def softmax(self, z):
        return torch.exp(z) / torch.sum(torch.exp(z), axis=1, keepdims=True)
    
    def forward(self, x):
        # automatically called within PyTorch when passing in the input
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    
class FFNN_REG(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(FFNN_REG, self).__init__()
        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        # Define input layer:
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        
    def sigmoid(self, z):
        #z = z.detach().numpy()
        return 1.0/(1.0 + torch.exp(-z))
        
    def softmax(self, z):
        #z = z.detach().numpy()
        e_z = torch.exp(z - torch.max(z))
        return e_z / e_z.sum(axis=0)
    
    def forward(self, x):
        # Apply regulaization here (Dropout)
        x = self.dropout(x)
        # automatically called within PyTorch when passing in the input
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    
    
    