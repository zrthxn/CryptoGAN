import torch
from torch import nn
from torch import functional as F


class LinearBlock(nn.Module):
  def __init__(self, inf, opf, act = nn.GELU):
    super(LinearBlock, self).__init__()
    self.fcl = nn.Linear(in_features=inf, out_features=opf)
    self.act = act()
  
  def forward(self, x):
    return self.act(self.fcl(x))


class Encoder(nn.Module):
  def __init__(self, shape):
    super(Encoder, self).__init__()

    self.layers = list()
    for i in range(len(shape) - 1):
      if not isinstance(shape[i], int):
        raise TypeError('Shape must be a list of type INT only')

      self.layers.append(LinearBlock(shape[i], shape[i+1], nn.Sigmoid))
    
    self.model = nn.Sequential(*self.layers)

  def forward(self, x):
    x = self.model(x)
    return x

