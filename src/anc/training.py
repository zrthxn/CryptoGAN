import itertools
import torch
from logging import info
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import defaults
from src.anc.datagen import KeyGenerator
from src.anc.datagen import PlainGenerator
from src.anc.model import KeyholderNetwork, AttackerNetwork


VERSION = '1.0'

class TrainingSession():
  def __init__(self, debug = False, 
      BLOCKSIZE = defaults["anc"]["blocksize"], 
      BATCHLEN = defaults["anc"]["batchlen"]):
    self.blocksize = BLOCKSIZE
    
    # Initialize Networks
    self.alice = KeyholderNetwork(BLOCKSIZE, name='Alice')
    self.bob = KeyholderNetwork(BLOCKSIZE, name='Bob')
    self.eve = AttackerNetwork(BLOCKSIZE, name='Eve')

    # CUDA
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else 'torch.Tensor')
    # info('Using device', str(device))

    self.alice.to(device)
    self.bob.to(device)
    self.eve.to(device)

    self.l1_loss = torch.nn.L1Loss()
    
    self.Key = KeyGenerator(BLOCKSIZE, BATCHLEN)
    self.Plain = PlainGenerator(BLOCKSIZE, BATCHLEN)

    self.debug = debug
    self.writer = SummaryWriter(f'training/anc_v{VERSION}') if not debug else None

  def log(self, *ip):
    if self.debug:
      print(*ip)

  # Training loop
  def train(self, BATCHES, EPOCHS):
    print(f'ANC Model v{VERSION}')
    ab_params = itertools.chain(self.alice.parameters(), self.bob.parameters())
    opt_alice = torch.optim.Adam(ab_params, lr=defaults["anc"]["alice_lr"])
    opt_eve = torch.optim.Adam(self.eve.parameters(), lr=defaults["anc"]["eve_lr"])

    alice_running_loss = []
    bob_running_loss = []
    eve_running_loss = []

    self.alice.train()
    self.bob.train()
    self.eve.train()

    # Training loop
    torch.autograd.set_detect_anomaly(True)
    
    print("Starting training loop")
    print(f'Training with {BATCHES} batches over {EPOCHS} epochs')

    for E in range(EPOCHS):
      print(f'Epoch {E + 1}/{EPOCHS}')
      train_turn = 0

      for B in tqdm(range(BATCHES)):
        PLAIN = self.Plain.single()        
        KEY = self.Key.single()
        
        K = torch.Tensor(KEY)
        P = torch.Tensor(PLAIN)

        cipher = self.alice(torch.cat([P, K], dim=1))
        Pb = self.bob(torch.cat([cipher, K], dim=1))
        Pe = self.eve(cipher)

        bob_reconst_loss = self.l1_loss(Pb, P)
        eve_reconst_loss = self.l1_loss(Pe, P)

        # Quadratic loss
        alice_loss = bob_reconst_loss - ((1 - eve_reconst_loss/(self.blocksize/2)) ** 2)

        if train_turn == 0:
          # Train Alice-Bob for 1 turn
          bob_reconst_loss.backward(retain_graph=True)
          alice_loss.backward(retain_graph=True)
          opt_alice.step()
        else:
          # Train Eve for 2 turns
          eve_reconst_loss.backward(retain_graph=True)
          opt_eve.step()

        train_turn += 1
        if train_turn >= 2:
          train_turn = 0

        alice_running_loss.append(alice_loss.item())
        bob_running_loss.append(bob_reconst_loss.item())
        eve_running_loss.append(eve_reconst_loss.item())

        if not self.debug:
          self.writer.add_scalar('Training Loss', bob_reconst_loss.item(), (E * BATCHES) + B)
          self.writer.add_scalar('Adversary Loss', eve_reconst_loss.item(), (E * BATCHES)  + B)
          self.writer.close()

    self.log('Finished Training')
    return (self.alice, self.bob, self.eve), (alice_running_loss, bob_running_loss, eve_running_loss)
