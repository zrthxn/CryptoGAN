"""
Secure Neural Cryptography setup using GANs as described in the Coutinho Paper.
To verify improvements on the previous design and make our own.

Coutinho, Murilo & Albuquerque, Robson & Borges, Fábio & García Villalba, et al (2018). 
Learning Perfectly Secure Cryptography to Protect Communications with 
Adversarial Neural Cryptography. Sensors. 18. 10.3390/s18051306.
"""

# %%
# Imports Section
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

VERSION = 25

# %%
# Define networks
VERSION += 1

class KeyholderNetwork(nn.Module):
  def __init__(self, blocksize):
    super(KeyholderNetwork, self).__init__()
    
    self.blocksize = blocksize

    # Entry layer verifies the size of inputs before proceeding
    self.entry = nn.Identity(blocksize * 2)

    self.fc1 = nn.Linear(in_features=blocksize*2, out_features=blocksize*4)
    self.fc2 = nn.Linear(in_features=blocksize*4, out_features=blocksize*2)
    self.fc3 = nn.Linear(in_features=blocksize * 2, out_features=blocksize)
    
    # self.norm = nn.BatchNorm1d(num_features=blocksize)
    
    # self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=2)
    # self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
    # self.conv3 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
    # self.conv4 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1)
 
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
    inputs = torch.acos(1 - torch.mul(inputs, 2)) / 4

    # inputs = torch.sigmoid(inputs)
    # inputs = F.hardsigmoid(inputs)

    inputs = self.fc1(inputs)
    inputs = torch.relu(inputs)

    inputs = self.fc2(inputs)
    inputs = torch.relu(inputs)
    
    inputs = self.fc3(inputs)
    inputs = torch.sigmoid(inputs)
    
    # f* = [1 - cos(a)]/2
    inputs = torch.div(1 - torch.cos(inputs), 2)
    
    # inputs = self.norm(inputs)

    # return inputs #.view(self.blocksize)
    return inputs



class AttackerNetwork(nn.Module):
  def __init__(self, blocksize):
    super(AttackerNetwork, self).__init__()
    
    self.blocksize = blocksize
    self.entry = nn.Identity(blocksize * 3)

    self.fc1 = nn.Linear(in_features=blocksize * 3, out_features=blocksize * 4)
    self.fc2 = nn.Linear(in_features=blocksize * 4, out_features=blocksize)
    self.fc3 = nn.Linear(in_features=blocksize, out_features=2)

  # def free_energy(self, v):
  #   vbias_term = v.mv(self.v_bias)
  #   wx_b = F.linear(v,self.W,self.h_bias)
  #   zr = torch. Variable(torch.zeros(wx_b.size()))
  #   mask = torch.max(zr, wx_b)
  #   hidden_term = (((wx_b - mask).exp() + (-mask).exp()).log() + (mask)).sum(1)
  #   return (-hidden_term - vbias_term).mean()

  def forward(self, inputs):
    inputs = self.entry(inputs)
    
    # f = arccos(1-2b)
    inputs = torch.acos(1 - torch.mul(inputs, 2))

    inputs = self.fc1(inputs)
    inputs = torch.sigmoid(inputs)

    # f* = [1 - cos(a)]/2
    inputs = torch.div(1 - torch.cos(inputs), 2)

    inputs = self.fc2(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.fc3(inputs)

    inputs = torch.softmax(inputs, dim=0)

    return inputs


# %%
# Data and Proprocessing

BLOCKSIZE = 16
EPOCHS = 10
BATCHES = 1024 #* 16
BATCHLEN = 16

KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]
PLAIN = [[
    [random.randint(0, 1) for x in range(BLOCKSIZE)],
    [random.randint(0, 1) for y in range(BLOCKSIZE)]
    ] for i in range(BATCHLEN)]

def recalc_key():
  KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]

def recalc_plain():
  PLAIN = [[
      [random.randint(0, 1) for x in range(BLOCKSIZE)],
      [random.randint(0, 1) for y in range(BLOCKSIZE)]
    ] for i in range(BATCHLEN)]

sns.set()
writer = SummaryWriter(f'training/cryptonet_vL{VERSION}')

# %%
# Initialize Networks

alice = KeyholderNetwork(BLOCKSIZE)
bob = KeyholderNetwork(BLOCKSIZE)
eve = AttackerNetwork(BLOCKSIZE)

# dist = nn.L1Loss()
dist = nn.MSELoss()
# dist = nn.CrossEntropyLoss()
# dist = nn.BCELoss()

opt_bob = torch.optim.Adam(bob.parameters(), lr=8e-3, weight_decay=1e-5)
opt_eve = torch.optim.Adam(eve.parameters(), lr=2e-3, weight_decay=1e-5)

graph_ip = torch.cat([torch.Tensor(PLAIN[0][0]), torch.Tensor(KEY)], dim=0).unsqueeze(0)
writer.add_graph(alice, graph_ip)
writer.close() 

def trendline(data, deg=1):
  for _ in range(deg):
    last = data[0]
    trend = []
    for x in data:
      trend.append((x+last)/2)
      last = x
    
    data = trend
 
  return trend

# %%
# Training loop

alice_running_loss = []
bob_running_loss = []
eve_running_loss = []

bob_bits_err = []
eve_bits_err = []

# Hyperparams
BETA = 1.0
GAMMA = 1.2
OMEGA = 0.75
DECISION_BOUNDARY = 0.5

print(f'Model v{VERSION}')
print(f'Training with {BATCHES * BATCHLEN} samples over {EPOCHS} epochs')

alice.train()
bob.train()
eve.train()

STOP = False
for E in range(EPOCHS):
  print(f'Epoch {E + 1}/{EPOCHS}')
  K = torch.Tensor(KEY)

  for B in range(BATCHES):
    # opt_alice.zero_grad()
    opt_bob.zero_grad()
    opt_eve.zero_grad()

    for X in PLAIN:
      P0 = torch.Tensor(X[0])
      P1 = torch.Tensor(X[1])

      R = random.randint(0, 1)
      P = torch.Tensor(X[R])
      
      C = alice(torch.cat([P, K], dim=0))
      Pb = bob(torch.cat([C, K], dim=0))
      
      if torch.isnan(C[0]):
        raise OverflowError(f'[BATCH {B}] {len(alice_running_loss)}: Alice Exploding Gradient')

      if torch.isnan(Pb[0][0]):
        raise OverflowError(f'[BATCH {B}] {len(bob_running_loss)}: Bob Exploding Gradient')

      C.detach()
      Re = eve(torch.cat([P0, P1, C], dim=0))

      if torch.isnan(Re[0][0]):
        raise OverflowError(f'[BATCH {B}] {len(eve_running_loss)}: Eve Exploding Gradient')

      bob_err = 0
      for b in range(BLOCKSIZE):
        if (P[b] == 0 and Pb[b] >= DECISION_BOUNDARY):
          bob_err += 1
        if (P[b] == 1 and Pb[b] < DECISION_BOUNDARY):
          bob_err += 1

      bob_bits_err.append(bob_err)
      
      eve_recogni_loss = dist(Re, torch.Tensor([1 - R, R]))

      alice_bobrc_loss = dist(Pb, P)
      # alice_bobrc_loss = BETA * dist(Pb, P) - GAMMA * dist(Re, torch.Tensor([1 - R, R])) - OMEGA * dist(P, C) 

      bob_running_loss.append(alice_bobrc_loss.item())
      eve_running_loss.append(eve_recogni_loss.item())

      alice_bobrc_loss.backward(retain_graph=True)
      eve_recogni_loss.backward(retain_graph=True)

      torch.nn.utils.clip_grad_norm_(bob.parameters(), 4.0)
      torch.nn.utils.clip_grad_norm_(eve.parameters(), 4.0)

      opt_bob.step()
      opt_eve.step()

    # recalc_plain()

    # # Stop when bits error is consistently zero for 1 batch
    # if bob_bits_err[-BATCHLEN:] == [0 for _ in range(BATCHLEN)]:
    #   break
    
    if STOP:
      break

    writer.add_scalar('Training Loss', alice_bobrc_loss.item(), len(bob_running_loss))
    writer.add_scalar('Adversary Loss', eve_recogni_loss.item(), len(eve_running_loss))
    writer.add_scalar('Bit Error', np.array([bob_err]), len(bob_bits_err))
    writer.close()
        
    for param in alice.parameters():
      if param.grad is not None:
        writer.add_histogram('Gradient', param.grad, len(bob_running_loss))
  
    writer.close()

  # recalc_key()

  # Stop when bit error is consistently zero for 3 Batches
  if bob_bits_err[-3 * BATCHLEN:] == [0 for _ in range(3 * BATCHLEN)]:
    STOP = True
  
  if STOP:
    break

print('Finished Training')


# %%
# Evaluation Plots
sns.set_style('whitegrid')
sf = 999 #min(int(BATCHES * EPOCHS/10), 50)

TITLE_TAG = f'{BLOCKSIZE} bits, {dist}, B={BETA} G={GAMMA} W={OMEGA}'
FILE_TAG = f'{EPOCHS}E{BATCHES}x{BATCHLEN}v{VERSION}'

SAVEPLOT = False
# Turn this line on and off to control plot saving
# SAVEPLOT = True

# plt.plot(trendline(alice_running_loss[:1000], sf))
plt.plot(trendline(bob_running_loss, sf))
plt.plot(trendline(eve_running_loss, sf))
plt.legend(['Bob', 'Eve'], loc='upper right')
# plt.legend(['Alice', 'Bob', 'Eve'], loc='upper right')
# plt.xlim(len(alice_running_loss) - 1000, len(alice_running_loss))
plt.xlabel('Samples')
plt.ylabel(f'Loss (SF {sf})')
plt.title(f'Training Loss - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/loss_{FILE_TAG}.png', dpi=400)
plt.show()

plt.plot(bob_bits_err, color='red')
plt.plot(trendline(bob_bits_err, sf), color='black')
plt.legend(['Actual', 'Trend'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Bit error (SF {sf})')
plt.title(f'Bit Error - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/error_{FILE_TAG}.png', dpi=400)
plt.show()
