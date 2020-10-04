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
import numpy as np
import wandb

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

VERSION = 22

# wbrun = wandb.init(project="cryptonet")

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
    
    self.norm = nn.BatchNorm1d(num_features=blocksize)
    
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

    inputs = torch.sigmoid(inputs)
    # inputs = F.hardsigmoid(inputs)

    # inputs = self.fc1(inputs)
    # inputs = torch.relu(inputs)

    # inputs = self.fc2(inputs)
    # inputs = torch.relu(inputs)
    
    inputs = self.fc3(inputs)
    inputs = torch.sigmoid(inputs)
    
    # f* = [1 - cos(a)]/2
    inputs = torch.div(1 - torch.cos(inputs), 2)
    
    inputs = self.norm(inputs)

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

  def free_energy(self, v):
    vbias_term = v.mv(self.v_bias)
    wx_b = F.linear(v,self.W,self.h_bias)
    zr = torch. Variable(torch.zeros(wx_b.size()))
    mask = torch.max(zr, wx_b)
    hidden_term = (((wx_b - mask).exp() + (-mask).exp()).log() + (mask)).sum(1)
    return (-hidden_term - vbias_term).mean()

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
EPOCHS = 4
BATCHES = 256
BATCHLEN = 64

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

# %%
# Initialize Networks

alice = KeyholderNetwork(BLOCKSIZE)
bob = KeyholderNetwork(BLOCKSIZE)
eve = AttackerNetwork(BLOCKSIZE)

dist = nn.L1Loss()
# dist = nn.MSELoss()
# dist = nn.CrossEntropyLoss()
# dist = nn.BCELoss()

# opt_alice = torch.optim.SGD(alice.parameters(), lr=8e-4, momentum=1e-1)
# opt_bob = torch.optim.SGD(bob.parameters(), lr=8e-4, momentum=1e-1)
# opt_eve = torch.optim.SGD(eve.parameters(), lr=2e-4, momentum=1e-1)
opt_alice = torch.optim.Adam(alice.parameters(), lr=8e-4, weight_decay=1e-5)
opt_bob = torch.optim.Adam(bob.parameters(), lr=8e-4, weight_decay=1e-5)
opt_eve = torch.optim.Adam(eve.parameters(), lr=2e-4, weight_decay=1e-5)

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

alice_grad = []
bob_grad = []
eve_grad = []

bob_bits_err = []
eve_bits_err = []

# Hyperparams
BETA = 1.0
GAMMA = 1.2
OMEGA = 1.5
DECISION_BOUNDARY = 0.5

print(f'Model v{VERSION}')
print(f'Training with {BATCHES * BATCHLEN} samples over {EPOCHS} epochs')

opt_alice.zero_grad()
opt_bob.zero_grad()
opt_eve.zero_grad()

alice.train()
bob.train()
eve.train()

# wbrun.watch(alice, log='all')

STOP = False
for E in range(EPOCHS):
  print(f'Epoch {E + 1}/{EPOCHS}')
  K = torch.Tensor(KEY)

  for B in range(BATCHES):

    P0 = torch.Tensor([P[0] for P in PLAIN])
    P1 = torch.Tensor([P[1] for P in PLAIN])

    R = torch.randint(2, (BATCHLEN,))
    P = torch.cat([s.unsqueeze(dim=0) for s in [torch.Tensor(PLAIN[r][R[r].item()]) for r in range(BATCHLEN)]], dim=0)
    
    Q = torch.cat([s.unsqueeze(dim=0) for s in [torch.cat([p, K], dim=0) for p in P]], dim=0)
    C = alice(Q)
    
    if torch.isnan(C[0][0]):
      raise OverflowError(f'[BATCH {B}] {len(alice_running_loss)}: Exploding Gradient')

    Db = torch.cat([s.unsqueeze(dim=0) for s in [torch.cat([c, K], dim=0) for c in C]], dim=0)
    Pb = bob(Db)

    if torch.isnan(Pb[0][0]):
      raise OverflowError(f'[BATCH {B}] {len(bob_running_loss)}: Exploding Gradient')

    De = torch.cat([s.unsqueeze(dim=0) for s in [torch.cat([P0[r], P1[r], C[r]], dim=0) for r in range(BATCHLEN)]], dim=0)
    Re = eve(De)

    if torch.isnan(Re[0][0]):
      raise OverflowError(f'[BATCH {B}] {len(eve_running_loss)}: Exploding Gradient')

    bob_err = 0
    for x in range(BATCHLEN):
      for b in range(BLOCKSIZE):
        if (P[x][b] == 0 and Pb[x][b] >= DECISION_BOUNDARY):
          bob_err += 1
        if (P[x][b] == 1 and Pb[x][b] < DECISION_BOUNDARY):
          bob_err += 1

    bob_bits_err.append(bob_err)
    
    bob_reconst_loss = dist(Pb, P)
    eve_recogni_loss = dist(Re, torch.cat([torch.Tensor([1 - r, r]).unsqueeze(dim=0) for r in R], dim=0))

    # alice_loss = (BETA*bob_reconst_loss) - (OMEGA*dist(P, C)) - (GAMMA*eve_recogni_loss)
    # alice_loss = bob_reconst_loss - eve_recogni_loss #- dist(P, C)

    # alice_running_loss.append(alice_loss.item())
    bob_running_loss.append(bob_reconst_loss.item())
    eve_running_loss.append(eve_recogni_loss.item())

    bob_reconst_loss.backward(retain_graph=True)
    eve_recogni_loss.backward(retain_graph=True)
    # alice_loss.backward(retain_graph=True)

    for param in alice.parameters():
      alice_grad.append(param.grad)

    torch.nn.utils.clip_grad_norm_(alice.parameters(), 4.0)
    torch.nn.utils.clip_grad_norm_(bob.parameters(), 4.0)
    torch.nn.utils.clip_grad_norm_(eve.parameters(), 4.0)

    # opt_alice.step()
    opt_bob.step()
    opt_eve.step()

    recalc_plain()

    # # Stop when bits error is consistently zero for 1 batch
    # if bob_bits_err[-BATCHLEN:] == [0 for _ in range(BATCHLEN)]:
    #   break
    
    if STOP:
      break

  recalc_key()

  # Stop when bit error is consistently zero for 3 Batches
  if bob_bits_err[-3 * BATCHLEN:] == [0 for _ in range(3 * BATCHLEN)]:
    STOP = True
  
  if STOP:
    break

print('Finished Training')


# %%
# Evaluation Plots
sns.set_style('whitegrid')
sf = min(int(BATCHES * EPOCHS/10), 50)

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


# %%
