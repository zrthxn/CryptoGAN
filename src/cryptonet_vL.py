"""
Coutinho, Murilo & Albuquerque, Robson & Borges, Fábio & García Villalba, et al (2018). 
Learning Perfectly Secure Cryptography to Protect Communications with 
Adversarial Neural Cryptography. Sensors. 18. 10.3390/s18051306.
"""

# %%
# Imports Section
import math
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from mname import modelname

VERSION = 64

DEBUGGING = False

def debug(*ip):
  if DEBUGGING:
    print(*ip)

# %%
# Define networks
VERSION += 1

class KeyholderNetwork(nn.Module):
  def __init__(self, blocksize):
    super(KeyholderNetwork, self).__init__()
    
    self.blocksize = blocksize

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

    inputs = inputs.unsqueeze(dim=0).unsqueeze(dim=0)
    debug(inputs)
    
    inputs = self.conv1(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.conv2(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.conv3(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.conv4(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = inputs.view(self.blocksize)

    inputs = self.fc3(inputs)
    inputs = torch.relu(inputs)

    debug(inputs)

    # f* = [1 - cos(a)]/2
    # inputs = torch.div(1 - torch.cos(inputs), 2)
    # inputs = torch.div(1 - inputs, 2)
    inputs = F.hardsigmoid(torch.mul(inputs, 10) - 5)

    return inputs


class AttackerNetwork(nn.Module):
  def __init__(self, blocksize):
    super(AttackerNetwork, self).__init__()
    
    self.blocksize = blocksize
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
    # inputs = F.hardsigmoid(torch.mul(inputs, 10) - 5)

    return inputs

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm2d") != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)

# %%
# Data and Proprocessing
BLOCKSIZE = 4
EPOCHS = 16
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

sns.set()
writer = None

if not DEBUGGING:
  writer = SummaryWriter(f'training/cryptonet_vL{VERSION}')

# %%
# Initialize Networks
alice = KeyholderNetwork(BLOCKSIZE)
bob = KeyholderNetwork(BLOCKSIZE)
eve = AttackerNetwork(BLOCKSIZE)

# Initialize weights
alice.apply(weights_init_normal)
bob.apply(weights_init_normal)
eve.apply(weights_init_normal)

# CUDA
cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
if cuda:
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

alice.to(device)
bob.to(device)
eve.to(device)

# dist = nn.L1Loss()
dist = nn.MSELoss()
# dist = nn.CrossEntropyLoss()
# dist = nn.BCELoss()

ab_params = itertools.chain(alice.parameters(), bob.parameters())
opt_alice_bob = torch.optim.Adam(ab_params, lr=1e-3, weight_decay=1e-5)

opt_eve = torch.optim.Adam(eve.parameters(), lr=2e-4)

if not DEBUGGING:
  graph_ip = torch.cat([torch.Tensor(PLAIN[0][0]), torch.Tensor(KEY)], dim=0)
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

bob_bits_acc = []

# Hyperparams
BETA = 1.0
GAMMA = 1.2
OMEGA = 0.75
DECISION_MARGIN = 0.1

print(f'Model v{VERSION}')
print(f'Training with {BATCHES * BATCHLEN} samples over {EPOCHS} epochs')

alice.train()
bob.train()
eve.train()

torch.autograd.set_detect_anomaly(True)

STOP = False
for E in range(EPOCHS):
  print(f'Epoch {E + 1}/{EPOCHS}')
  K = torch.Tensor(KEY)

  for B in range(BATCHES):
    for X in PLAIN:
      P0 = torch.Tensor(X[0])
      P1 = torch.Tensor(X[1])

      R = random.randint(0, 1)
      P = torch.Tensor(X[R])
      # debug('PLAIN', P)
            
      C = alice(torch.cat([P, K], dim=0))
      # debug('CIPHR', C)

      Pb = bob(torch.cat([C, K], dim=0))
      # debug('DCRPT', Pb)

      Re = eve(torch.cat([P0, P1, C.detach()], dim=0))
      # debug('EVERE', Re)
      
      # Loss and BackProp
      eve_adv_loss = dist(Re, torch.Tensor([1 - R, R]))
      bob_dec_loss = dist(Pb, P) + torch.square(1 - eve_adv_loss)
      
      opt_alice_bob.zero_grad()
      bob_dec_loss.backward(retain_graph=True)
      opt_alice_bob.step()

      opt_eve.zero_grad()
      eve_adv_loss.backward(retain_graph=True)
      opt_eve.step()

      # torch.nn.utils.clip_grad_norm_(alice.parameters(), 4.0)
      # torch.nn.utils.clip_grad_norm_(bob.parameters(), 4.0)
      # torch.nn.utils.clip_grad_norm_(eve.parameters(), 4.0)

      bob_acc = 0
      for b in range(BLOCKSIZE):
        if torch.abs(torch.round(Pb[b] - DECISION_MARGIN)) == P[b]:
          bob_acc += (1/BLOCKSIZE)

      bob_bits_acc.append(bob_acc)
      bob_running_loss.append(bob_dec_loss.item())
      eve_running_loss.append(eve_adv_loss.item())

      if STOP:
        break

    # recalc_plain()

    if not DEBUGGING:
      writer.add_scalar('Training Loss', bob_dec_loss.item(), (E * BATCHES) + B)
      writer.add_scalar('Bit Accuracy', torch.Tensor([bob_acc]), (E * BATCHES) + B)
      writer.add_scalar('Adversary Loss', eve_adv_loss.item(), (E * BATCHES)  + B)
      writer.close()
    
    if STOP:
      break
  
  # recalc_key()
  
  if STOP:
    break

print('Finished Training')


# %%
# Evaluation Plots
sns.set_style('whitegrid')
sf = 5000 #min(int(BATCHES * EPOCHS/10), 50)

TITLE_TAG = f'{BLOCKSIZE} bits, {dist}, B={BETA} G={GAMMA} W={OMEGA}'
FILE_TAG = f'{EPOCHS}E{BATCHES}x{BATCHLEN}v{VERSION}'

SAVEPLOT = False
# Turn this line on and off to control plot saving
# SAVEPLOT = True

# plt.plot(trendline(alice_running_loss[:1000], sf))
plt.plot(bob_running_loss)
plt.plot(trendline(bob_running_loss, sf), color='black')
# plt.plot(trendline(eve_running_loss, sf))
plt.legend(['Bob', 'Eve'], loc='upper right')
# plt.legend(['Alice', 'Bob', 'Eve'], loc='upper right')
# plt.xlim(len(alice_running_loss) - 1000, len(alice_running_loss))
plt.xlabel('Samples')
plt.ylabel(f'Loss (SF {sf})')
plt.title(f'Training Loss - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/loss_{FILE_TAG}.png', dpi=400)
plt.show()

plt.plot(bob_bits_acc, color='red')
plt.plot(trendline(bob_bits_acc, sf), color='black')
plt.legend(['Actual', 'Trend'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Bit error (SF {sf})')
plt.title(f'Bit Error - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/error_{FILE_TAG}.png', dpi=400)
plt.show()


# %%
# Evaluation Plots



# %%
# Save Models

torch.save(alice, '../models/cryptonet/' + modelname('Alice', f'{BLOCKSIZE}x3', f'v{VERSION}'))
torch.save(bob, '../models/cryptonet/' + modelname('Bob', f'{BLOCKSIZE}x3', f'v{VERSION}'))
torch.save(eve, '../models/cryptonet/' + modelname('Eve', f'{BLOCKSIZE}x3', f'v{VERSION}'))
