"""
Abadi, M.; Andersen, D.G.; ICLR 2017
Learning to protect communications with adversarial neural cryptography. 
arXiv 2016, arXiv:1610.06918.
"""

# Imports Section
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VERSION = 15

# Define networks
class KeyholderNetwork(nn.Module):
  def __init__(self, blocksize):
    super(KeyholderNetwork, self).__init__()
    
    self.blocksize = blocksize
    self.entry = nn.Identity(blocksize * 2)

    self.fc1 = nn.Linear(in_features=blocksize * 2, out_features=blocksize * 2)
    
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=2)
    self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
    self.conv3 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
    self.conv4 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1)

  def forward(self, inputs):    
    inputs = self.entry(inputs)

    inputs = self.fc1(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = inputs.unsqueeze(0).unsqueeze(0)

    inputs = self.conv1(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.conv2(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.conv3(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.conv4(inputs)
    inputs = F.hardsigmoid(inputs)

    return inputs.view(self.blocksize)

class AttackerNetwork(nn.Module):
  def __init__(self, blocksize):
    super(AttackerNetwork, self).__init__()
    
    self.blocksize = blocksize
    self.entry = nn.Identity(blocksize)

    self.fc1 = nn.Linear(in_features=blocksize, out_features=blocksize * 2)
    
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=2)
    self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
    self.conv3 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
    self.conv4 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1)

  def forward(self, inputs):    
    inputs = self.entry(inputs)

    inputs = self.fc1(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = inputs.unsqueeze(0).unsqueeze(0)
    
    inputs = self.conv1(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.conv2(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.conv3(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.conv4(inputs)
    inputs = F.hardsigmoid(inputs)

    return inputs.view(self.blocksize)
