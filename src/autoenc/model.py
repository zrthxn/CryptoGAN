import torch
from torch import nn
from torch import functional as F

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.model = nn.Sequential([
      *self.conv_block()
    ])

  def conv_block(self):
    return [
      nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2), stride=1),
      nn.MaxPool2d(kernel_size=(2,2), stride=1),
      # activation
    ]

  def forward(self, x):
    x = self.model(x)
    return x

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.model = nn.Sequential([
      nn.Linear(in_features=8, out_features=8)
    ])
  
  def forward(self, x):
    x = self.model(x)
    return x