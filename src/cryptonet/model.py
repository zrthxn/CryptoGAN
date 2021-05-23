"""
Coutinho, Murilo & Albuquerque, Robson & Borges, Fábio & García Villalba, et al (2018). 
Learning Perfectly Secure Cryptography to Protect Communications with 
Adversarial Neural Cryptography. Sensors. 18. 10.3390/s18051306.
"""

# Imports Section
import math
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VERSION = 64

DEBUGGING = False

def debug(*ip):
  if DEBUGGING:
    print(*ip)

# Define networks
VERSION += 1

class KeyholderNetwork(nn.Module):
  def __init__(self, blocksize, name = None):
    super(KeyholderNetwork, self).__init__()
    
    self.blocksize = blocksize
    self.name = name if name != None else 'KeyholderNetwork'

    # Entry layer verifies the size of inputs before proceeding
    self.entry = nn.Identity(blocksize * 2)

    self.fc1 = nn.Linear(in_features=blocksize * 2, out_features=blocksize * 4)
    self.fc2 = nn.Linear(in_features=blocksize * 4, out_features=blocksize * 2)
    self.fc3 = nn.Linear(in_features=blocksize, out_features=blocksize)
    
    # self.norm = nn.BatchNorm1d(num_features=blocksize)
    
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=2)
    self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
    self.conv3 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
    self.conv4 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1)
 
  def squash(self, input_):
    squared_norm = (input_ ** 2).sum(-1, keepdim=True)
    denom = ((1. + squared_norm) * torch.sqrt(squared_norm))
    if torch.isinf(denom).sum().item() > 0:
      output_ = input_ / torch.sqrt(squared_norm)
    else:
      output_ = squared_norm * input_ / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_

  def forward(self, inputs):    
    inputs = self.entry(inputs)

    # f = arccos(1-2b)
    # inputs = torch.acos(1 - torch.mul(inputs, 2))
    debug(inputs)

    inputs = self.fc1(inputs)
    inputs = torch.relu(inputs)

    inputs = self.fc2(inputs)
    inputs = torch.relu(inputs)

    inputs = inputs.unsqueeze(dim=1)
    debug(inputs)
    
    inputs = self.conv1(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.conv2(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.conv3(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.conv4(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = inputs.view((-1, self.blocksize))

    inputs = self.fc3(inputs)
    inputs = torch.relu(inputs)

    debug(inputs)

    # f* = [1 - cos(a)]/2
    # inputs = torch.div(1 - torch.cos(inputs), 2)
    # inputs = torch.div(1 - inputs, 2)
    inputs = F.hardsigmoid(torch.mul(inputs, 10) - 5)

    return inputs


class AttackerNetwork(nn.Module):
  def __init__(self, blocksize, name = None):
    super(AttackerNetwork, self).__init__()
    
    self.blocksize = blocksize
    self.name = name if name != None else 'AttackerNetwork'
    self.entry = nn.Identity(blocksize * 3)

    self.fc1 = nn.Linear(in_features=blocksize * 3, out_features=blocksize * 6)
    self.fc2 = nn.Linear(in_features=blocksize * 6, out_features=blocksize * 4)
    self.fc3 = nn.Linear(in_features=blocksize * 4, out_features=blocksize * 2)
    self.fc4 = nn.Linear(in_features=blocksize * 2, out_features=blocksize)
    self.fc5 = nn.Linear(in_features=blocksize, out_features=2)

  def forward(self, inputs):
    inputs = self.entry(inputs)
    
    # f = arccos(1-2b)
    # inputs = torch.acos(1 - torch.mul(inputs, 2))

    inputs = self.fc1(inputs)
    inputs = torch.relu(inputs)

    inputs = self.fc2(inputs)
    inputs = torch.relu(inputs)

    inputs = self.fc3(inputs)
    inputs = torch.relu(inputs)

    inputs = self.fc4(inputs)
    inputs = torch.relu(inputs)

    inputs = self.fc5(inputs)
    inputs = torch.softmax(inputs, dim=0)

    # f* = [1 - cos(a)]/2
    # inputs = torch.div(1 - torch.cos(inputs), 2)
    inputs = F.hardsigmoid(torch.mul(inputs, 10) - 5)

    return inputs

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm2d") != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)
