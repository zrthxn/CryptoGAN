"""
  ANC Test setup using GANs as described in the Abadi Paper

  Learning to protect communications with adversarial neural cryptography. 
  Abadi, M.; Andersen, D.G.; arXiv 2016, arXiv:1610.06918.
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

VERSION = 15

# %%
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


# %%
# Data and preprocessing

BLOCKSIZE = 16

KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]
PLAIN = [[random.randint(0, 1) for x in range(BLOCKSIZE)] for i in range(2048)]

BATCHES = 12

def recalc_key():
  KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]

def recalc_plain():
  PLAIN = [[random.randint(0, 1) for x in range(BLOCKSIZE)] for i in range(2048)]

  
# %%
# Initialize Networks

alice = KeyholderNetwork(BLOCKSIZE)
bob = KeyholderNetwork(BLOCKSIZE)
eve = AttackerNetwork(BLOCKSIZE)

# lossfn = nn.L1Loss()
# lossfn = nn.MSELoss()
lossfn = nn.BCELoss()

opt_alice = torch.optim.Adam(alice.parameters(), lr=0.0008)
opt_bob = torch.optim.Adam(bob.parameters(), lr=0.0008)
opt_eve = torch.optim.Adam(eve.parameters(), lr=0.0008)

VERSION += 1

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
GAMMA = 1.0
DECISION_BOUNDARY = 0.5

for E in range(BATCHES):
  for P in PLAIN:
    P = torch.Tensor(P)
    K = torch.Tensor(KEY)

    cipher = alice(torch.cat([P, K], dim=0))
    cipher.detach()

    Pb = bob(torch.cat([cipher, K], dim=0))
    Pe = eve(cipher)

    bob_err = 0
    eve_err = 0
    for b in range(BLOCKSIZE):
      if (P[b] == 0 and Pb[b] >= DECISION_BOUNDARY):
        bob_err += 1
      if (P[b] == 1 and Pb[b] < DECISION_BOUNDARY):
        bob_err += 1

      if (P[b] == 0 and Pe[b] >= DECISION_BOUNDARY):
        eve_err += 1
      if (P[b] == 1 and Pe[b] < DECISION_BOUNDARY):
        eve_err += 1

    bob_bits_err.append(bob_err)
    eve_bits_err.append(eve_err)

    bob_reconst_loss = lossfn(Pb, P)
    eve_reconst_loss = lossfn(Pe, P)

    # Linear loss
    # alice_loss = (BETA * bob_reconst_loss) - (GAMMA * eve_reconst_loss)

    # Quad loss
    alice_loss = bob_reconst_loss - (((BLOCKSIZE/2) - eve_reconst_loss)**2/(BLOCKSIZE/2)**2)

    bob_reconst_loss.backward(retain_graph=True)
    eve_reconst_loss.backward(retain_graph=True)
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
    
    # print('Alice', alice_loss.item(), #f'{delta_alice/alice_loss.item()}%',
    #   '\tBob', bob_reconst_loss.item(),# f'{delta_bob/bob_reconst_loss.item()}%',
    #   '\tEve', eve_reconst_loss.item(), #f'{delta_eve/eve_reconst_loss.item()}%'
    # )

    alice_running_loss.append(alice_loss.item())
    bob_running_loss.append(bob_reconst_loss.item())
    eve_running_loss.append(eve_reconst_loss.item())
    
    # Recalculate key after every 100 items
    # if (len(alice_running_loss) / 100 == 0):
    #   recalc_key()
    
    # break

  print(f'Finished Batch {E}')
  recalc_plain()
  recalc_key()

print('Finished Training')

# %%
# Evaluation Plots
sns.set_style('whitegrid')
sf = 1800

# fig, [ax, bx] = plt.subplot(1, 1)

TITLE_TAG = f'{BLOCKSIZE} bits, Quadratic BCE loss, B={BETA} G={GAMMA}'
FILE_TAG = f'{BATCHES}x{len(PLAIN)}v{VERSION}'

SAVEPLOT = False
# Turn this line on and off to control plot saving
SAVEPLOT = True

plt.plot(trendline(alice_running_loss, sf))
plt.plot(trendline(bob_running_loss, sf))
plt.plot(trendline(eve_running_loss, sf))
plt.legend(['Alice', 'Bob', 'Eve'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Loss Trend (Sf {sf})')
plt.title(f'Training - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/graphs/loss_{FILE_TAG}.png', dpi=400)
plt.show()

plt.plot(trendline(bob_bits_err, sf))
plt.plot(trendline(eve_bits_err, sf))
plt.legend(['Bob', 'Eve'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Bit error trend (Sf {sf})')
plt.title(f'Error - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/graphs/error_{FILE_TAG}.png', dpi=400)
plt.show()


bob_bits_err_test = []
eve_bits_err_test = []

recalc_plain()
recalc_key()

for P in PLAIN:
  P = torch.Tensor(P)
  K = torch.Tensor(KEY)

  cipher = alice(torch.cat([P, K], dim=0))
  cipher.detach()

  Pb = bob(torch.cat([cipher, K], dim=0))
  Pe = eve(cipher)

  bob_err = 0
  eve_err = 0
  for b in range(BLOCKSIZE):
    if (P[b] == 0 and Pb[b] >= DECISION_BOUNDARY):
      bob_err += 1
    if (P[b] == 1 and Pb[b] < DECISION_BOUNDARY):
      bob_err += 1

    if (P[b] == 0 and Pe[b] >= DECISION_BOUNDARY):
      eve_err += 1
    if (P[b] == 1 and Pe[b] < DECISION_BOUNDARY):
      eve_err += 1

  bob_bits_err_test.append(bob_err)
  eve_bits_err_test.append(eve_err)

plt.plot(trendline(bob_bits_err_test, sf))
plt.plot(trendline(eve_bits_err_test, sf))
plt.legend(['Bob', 'Eve'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Bit error trend (Sf {sf})')
plt.title(f'Val Error - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/graphs/val_error_{FILE_TAG}.png', dpi=400)
plt.show()

