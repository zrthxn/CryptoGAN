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

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

VERSION = 17

# %%
# Define networks
VERSION += 1

class KeyholderNetwork(nn.Module):
  def __init__(self, blocksize):
    super(KeyholderNetwork, self).__init__()
    
    self.blocksize = blocksize

    # Entry layer verifies the size of inputs before proceeding
    self.entry = nn.Identity(blocksize * 2)

    self.fc1 = nn.Linear(in_features=blocksize*2, out_features=blocksize*2)
    self.fc2 = nn.Linear(in_features=blocksize*2, out_features=blocksize)
    self.fc3 = nn.Linear(in_features=blocksize, out_features=blocksize)
    
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, stride=1, padding=2)
    self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
    self.conv3 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, stride=1)
    self.conv4 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, stride=1)
 
  def forward(self, inputs):    
    inputs = self.entry(inputs)
    inputs = torch.acos(1 - torch.mul(inputs, 2))

    inputs = self.fc1(inputs)
    inputs = torch.sigmoid(inputs)

    inputs = self.fc2(inputs)
    inputs = torch.sigmoid(inputs)
    
    inputs = self.fc3(inputs)
    inputs = torch.sigmoid(inputs)

    # inputs = inputs.unsqueeze(0).unsqueeze(0)

    # inputs = self.conv1(inputs)
    # inputs = torch.sigmoid(inputs)
    
    # inputs = self.conv2(inputs)
    # inputs = torch.sigmoid(inputs)

    # inputs = self.conv3(inputs)
    # inputs = torch.sigmoid(inputs)
    
    # inputs = self.conv4(inputs)
    # inputs = F.hardsigmoid(inputs)
    inputs = torch.div(1 - torch.cos(inputs), 2)

    return inputs.view(self.blocksize)



class AttackerNetwork(nn.Module):
  def __init__(self, blocksize):
    super(AttackerNetwork, self).__init__()
    
    self.blocksize = blocksize
    self.entry = nn.Identity(blocksize * 3)

    self.fc1 = nn.Linear(in_features=blocksize * 3, out_features=blocksize * 4)
    self.fc2 = nn.Linear(in_features=blocksize * 4, out_features=blocksize)
    self.fc3 = nn.Linear(in_features=blocksize, out_features=2)

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
EPOCHS = 1
BATCHES = 64
BATCHLEN = 256

KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]
PLAIN = [
  [
    [random.randint(0, 1) for x in range(BLOCKSIZE)],
    [random.randint(0, 1) for y in range(BLOCKSIZE)]
  ] for i in range(BATCHLEN)
]

def recalc_key():
  KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]

def recalc_plain():
  PLAIN = [
    [
      [random.randint(0, 1) for x in range(BLOCKSIZE)],
      [random.randint(0, 1) for y in range(BLOCKSIZE)]
    ] for i in range(BATCHLEN)
  ]

# %%
# Initialize Networks

alice = KeyholderNetwork(BLOCKSIZE)
bob = KeyholderNetwork(BLOCKSIZE)
eve = AttackerNetwork(BLOCKSIZE)

dist = nn.MSELoss()
# dist = nn.L1Loss()
# dist = nn.CrossEntropyLoss()
# dist = nn.BCELoss()

opt_alice = torch.optim.Adam(alice.parameters(), lr=0.0008)
opt_bob = torch.optim.Adam(bob.parameters(), lr=0.0008)
opt_eve = torch.optim.Adam(eve.parameters(), lr=0.0002)


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
BETA = 1.5
GAMMA = 1.0
OMEGA = 1.5
DECISION_BOUNDARY = 0.5

print(f'Model v{VERSION}')
print(f'Training with {BATCHES * BATCHLEN} samples over {EPOCHS} epochs')
for E in range(EPOCHS):
  print(f'Epoch {E + 1}/{EPOCHS}')
  for B in range(BATCHES):
    for P in PLAIN:
      K = torch.Tensor(KEY)

      P0 = torch.Tensor(P[0])
      P1 = torch.Tensor(P[1])

      R = random.randint(0, 1)
      Q = torch.Tensor(P[R])

      C = alice(torch.cat([Q, K], dim=0))

      Pb = bob(torch.cat([C, K], dim=0))
      Re = eve(torch.cat([P0, P1, C], dim=0))

      bob_err = 0
      for b in range(BLOCKSIZE):
        if (Q[b] == 0 and Pb[b] >= DECISION_BOUNDARY):
          bob_err += 1
        if (Q[b] == 1 and Pb[b] < DECISION_BOUNDARY):
          bob_err += 1

      bob_bits_err.append(bob_err)
      
      bob_reconst_loss = dist(Pb, Q) #- dist(Q, C)
      eve_recogni_loss = dist(Re, torch.Tensor([1 - R, R]))

      # Linear loss
      alice_loss = (BETA * bob_reconst_loss) - (OMEGA * dist(Q, C)) - (GAMMA * eve_recogni_loss)
      # alice_loss = BETA * dist(Pb, Q)

      bob_reconst_loss.backward(retain_graph=True)
      eve_recogni_loss.backward(retain_graph=True)
      alice_loss.backward(retain_graph=True)

      opt_alice.step()
      opt_bob.step()
      opt_eve.step()

      # delta_alice = alice_running_loss[-1] - alice_loss.item()
      # delta_bob = bob_running_loss[-1] - bob_reconst_loss.item()
      # delta_eve = eve_running_loss[-1] - eve_reconst_loss.item()

      # Stop when no/low avg change
      # if ((delta_alice + delta_bob + delta_eve)/3 <= 0.00005):
      #   print('--- Training Stalled ---')
      #   break

      alice_running_loss.append(alice_loss.item())
      bob_running_loss.append(bob_reconst_loss.item())
      eve_running_loss.append(eve_recogni_loss.item())
      
      # break

    # print(f'Finished Batch {B}')
    
    recalc_plain()
    recalc_key()

    # Stop when bits error is consistently zero for 1 batch
    if bob_bits_err[-BATCHLEN:] == [0 for _ in range(BATCHLEN)]:
      break

  # Stop when bits error is consistently zero for 3 Batches
  if bob_bits_err[-3*BATCHLEN:] == [0 for _ in range(3*BATCHLEN)]:
    break

print('Finished Training')


# %%
# Evaluation Plots
sns.set_style('whitegrid')
sf = min(int(BATCHES * BATCHLEN/10), 100)

TITLE_TAG = f'[No Adv] {BLOCKSIZE} bits, {dist}, B={BETA} G={GAMMA}'
FILE_TAG = f'{EPOCHS}E{BATCHES}x{BATCHLEN}v{VERSION}'

SAVEPLOT = False
# Turn this line on and off to control plot saving
SAVEPLOT = True

# plt.xkcd()
plt.plot(trendline(alice_running_loss[:2000], sf))
plt.plot(trendline(bob_running_loss[:2000], sf))
plt.plot(trendline(eve_running_loss[:2000], sf))
plt.legend(['Alice', 'Bob', 'Eve'], loc='upper right')
# plt.xlim(len(bob_running_loss) - 2000, len(bob_running_loss))
plt.xlabel('Samples')
plt.ylabel(f'Loss Trend (SF {sf})')
plt.title(f'Training Loss - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/loss_{FILE_TAG}.png', dpi=400)
plt.show()

# plt.plot(bob_bits_err)
plt.plot(trendline(bob_bits_err[:2000], sf), color='green')
plt.legend(['Trend'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Bit error trend (SF {sf})')
plt.title(f'Bit Error - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/error_{FILE_TAG}.png', dpi=400)
plt.show()

# %%
# Evaluation
# bob_bits_err_test = []

# recalc_plain()
# recalc_key()

# for P in PLAIN:
#   P = torch.Tensor(P)
#   K = torch.Tensor(KEY)

#   C = alice(torch.cat([P, K], dim=0))
#   C.detach()

#   Pb = bob(torch.cat([C, K], dim=0))

#   bob_err = 0
#   eve_err = 0
#   for b in range(BLOCKSIZE):
#     if (P[b] == 0 and Pb[b] >= DECISION_BOUNDARY):
#       bob_err += 1
#     if (P[b] == 1 and Pb[b] < DECISION_BOUNDARY):
#       bob_err += 1

#   bob_bits_err_test.append(bob_err)

# plt.plot(bob_bits_err_test)
# plt.plot(trendline(bob_bits_err_test, sf))
# plt.legend(['Bob', 'Eve'], loc='upper right')
# plt.xlabel('Samples')
# plt.ylabel(f'Bit error trend (Sf {sf})')
# plt.title(f'Val Error - {TITLE_TAG}')
# if SAVEPLOT:
#   plt.savefig(f'../models/graphs/val_error_{FILE_TAG}.png', dpi=400)
# plt.show()


# %%
