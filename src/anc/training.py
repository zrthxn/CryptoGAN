import itertools
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from anc.datagen import KeyGenerator as Key
from anc.datagen import PlainGenerator as Plain

from anc.model import KeyholderNetwork, AttackerNetwork

BLOCKSIZE = 12
BATCHLEN = 64

VERSION = 15

KeyGenerator = Key(BLOCKSIZE)
PlainGenerator = Plain(BLOCKSIZE, BATCHLEN)

class TrainingSession():
  def __init__(self, debug = False):
    # Initialize Networks
    self.alice = KeyholderNetwork(BLOCKSIZE)
    self.bob = KeyholderNetwork(BLOCKSIZE)
    self.eve = AttackerNetwork(BLOCKSIZE)

    # CUDA
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.Tensor')

    self.alice.to(device)
    self.bob.to(device)
    self.eve.to(device)

    self.lossfn = torch.nn.L1Loss()

    self.debug = debug
    self.writer = SummaryWriter(f'training/cryptonet_vL{VERSION}') if not debug else None

  def log(self, *ip):
    if self.debug:
      print(*ip)

  # Training loop
  def train(self, EPOCHS = 16, BATCHES = 256):
    KEY = KeyGenerator.batchgen()
    PLAIN = PlainGenerator.batchgen()

    ab_params = itertools.chain(self.alice.parameters(), self.bob.parameters())
    opt_alice = torch.optim.Adam(ab_params, lr=0.0008)
    opt_eve = torch.optim.Adam(self.eve.parameters(), lr=0.0008)

    alice_running_loss = []
    bob_running_loss = []
    eve_running_loss = []

    bob_bits_err = []
    eve_bits_err = []

    # Hyperparams
    BETA = 1.0
    GAMMA = 1.2
    DECISION_BOUNDARY = 0.5

    for E in range(BATCHES):
      for P in PLAIN:
        P = torch.Tensor(P)
        K = torch.Tensor(KEY)

        cipher = self.alice(torch.cat([P, K], dim=0))
        cipher.detach()

        Pb = self.bob(torch.cat([cipher, K], dim=0))
        Pe = self.eve(cipher)

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

        bob_reconst_loss = self.lossfn(Pb, P)
        eve_reconst_loss = self.lossfn(Pe, P)

        # Linear loss
        alice_loss = (BETA * bob_reconst_loss) - (GAMMA * eve_reconst_loss)

        # Quad loss
        # alice_loss = bob_reconst_loss - (((BLOCKSIZE/2) - eve_reconst_loss)**2/(BLOCKSIZE/2)**2)

        bob_reconst_loss.backward(retain_graph=True)
        eve_reconst_loss.backward(retain_graph=True)
        alice_loss.backward(retain_graph=True)

        opt_alice.step()
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

    self.log('Finished Training')
    return (alice_running_loss, bob_running_loss, eve_running_loss)
