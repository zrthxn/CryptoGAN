import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

from anc.datagen import KeyGenerator as Key
from anc.datagen import PlainGenerator as Plain

from anc.model import KeyholderNetwork, AttackerNetwork

from autoenc.datagen import ghetto_tqdm

VERSION = 1.0

class TrainingSession():
  def __init__(self, debug = False, BLOCKSIZE = 16, BATCHLEN = 64):
    self.blocksize = BLOCKSIZE
    
    # Initialize Networks
    self.alice = KeyholderNetwork("Alice", BLOCKSIZE)
    self.bob = KeyholderNetwork("Bob", BLOCKSIZE)
    self.eve = AttackerNetwork("Eve", BLOCKSIZE)

    # CUDA
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.Tensor')
    # print('Using device', device)

    # self.alice.to(device)
    # self.bob.to(device)
    # self.eve.to(device)

    self.lossfn = torch.nn.L1Loss()
    
    self.KeyGenerator = Key(BLOCKSIZE)
    self.PlainGenerator = Plain(BLOCKSIZE, BATCHLEN)

    self.debug = debug
    self.writer = SummaryWriter(f'training/anc_v{VERSION}') if not debug else None

  def log(self, *ip):
    if self.debug:
      print(*ip)

  # Training loop
  def train(self, BATCHES, EPOCHS):
    ab_params = itertools.chain(self.alice.parameters(), self.bob.parameters())
    opt_alice = torch.optim.Adam(ab_params, lr=0.0008)
    opt_eve = torch.optim.Adam(self.eve.parameters(), lr=0.001)

    alice_running_loss = []
    bob_running_loss = []
    eve_running_loss = []
    
    KEYS = self.KeyGenerator.batchgen(BATCHES)
    PLAINS = self.PlainGenerator.batchgen(BATCHES)

    print(f'ANC Model v{VERSION}')
    print(f'Training with {BATCHES} batches over {EPOCHS} epochs')

    for E in range(EPOCHS):
      print(f'Epoch {E + 1}/{EPOCHS}')

      for B in range(BATCHES):
        PLAIN = torch.Tensor(PLAINS[B])
        KEY = torch.Tensor(KEYS[B])

        # print(ghetto_tqdm(B, BATCHES))

        for P in PLAIN:
          P = torch.Tensor(P)
          K = torch.Tensor(KEY)

          cipher = self.alice(torch.cat([P, K], dim=0))
          Pb = self.bob(torch.cat([cipher, K], dim=0))
          Pe = self.eve(cipher)

          bob_reconst_loss = self.lossfn(Pb, P)
          eve_reconst_loss = self.lossfn(Pe, P)

          # Linear loss
          alice_loss = bob_reconst_loss - eve_reconst_loss

          bob_reconst_loss.backward(retain_graph=True)
          eve_reconst_loss.backward(retain_graph=True)
          alice_loss.backward(retain_graph=True)

          opt_alice.step()
          opt_eve.step()

        alice_running_loss.append(alice_loss.item())
        bob_running_loss.append(bob_reconst_loss.item())
        eve_running_loss.append(eve_reconst_loss.item())

        if not self.debug:
          self.writer.add_scalar('Training Loss', bob_reconst_loss.item(), (E * BATCHES) + B)
          self.writer.add_scalar('Adversary Loss', eve_reconst_loss.item(), (E * BATCHES)  + B)
          self.writer.close()

    self.log('Finished Training')
    return (self.alice, self.bob, self.eve), (alice_running_loss, bob_running_loss, eve_running_loss)
