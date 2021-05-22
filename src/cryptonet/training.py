import torch
import itertools
import random
from logging import info
from datetime import datetime
from os import path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import defaults
from src.cryptonet.datagen import KeyGenerator
from src.cryptonet.datagen import PlainGenerator
from src.cryptonet.model import KeyholderNetwork, AttackerNetwork
from src.cryptonet.model import weights_init_normal


VERSION = '1.1'

class TrainingSession():
  def __init__(self, debug = False, 
      BLOCKSIZE = defaults["cryptonet"]["blocksize"], 
      BATCHLEN = defaults["cryptonet"]["batchlen"]):
    self.blocksize = BLOCKSIZE
    
    # Initialize Networks
    self.alice = KeyholderNetwork(BLOCKSIZE, name='Alice')
    self.bob = KeyholderNetwork(BLOCKSIZE, name='Bob')
    self.eve = AttackerNetwork(BLOCKSIZE, name='Eve')

    # Initialize weights
    self.alice.apply(weights_init_normal)
    self.bob.apply(weights_init_normal)
    self.eve.apply(weights_init_normal)

    # # CUDA
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # torch.set_default_tensor_type('torch.cuda.Tensor' if torch.cuda.is_available() else 'torch.Tensor')
    # info('Using device', str(self.device))

    self.alice.to(self.device)
    self.bob.to(self.device)
    self.eve.to(self.device)

    # self.lossfn = torch.nn.L1Loss()
    self.lossfn = torch.nn.MSELoss()
    # self.lossfn = torch.nn.CrossEntropyLoss()
    # self.lossfn = torch.nn.BCELoss()

    self.Key = KeyGenerator(BLOCKSIZE, BATCHLEN)
    self.Plain = PlainGenerator(BLOCKSIZE, BATCHLEN)

    self.debug = debug
    self.logdir = f'training/cryptonet_vL{VERSION}/'
    self.writer = SummaryWriter(log_dir=path.join(self.logdir, str(datetime.now()))) if not debug else None

  def log(self, *ip):
    if self.debug:
      print(*ip)

  def train(self, BATCHES = 256, EPOCHS = 16):
    print(f'Cryptonet Model v{VERSION}')
    ab_params = itertools.chain(self.alice.parameters(), self.bob.parameters())
    opt_alice_bob = torch.optim.Adam(ab_params, lr=defaults["cryptonet"]["alice_lr"], weight_decay=1e-5)
    opt_eve = torch.optim.Adam(self.eve.parameters(), lr=defaults["cryptonet"]["eve_lr"])

    alice_running_loss = []
    bob_running_loss = []
    eve_running_loss = []

    bob_bits_acc = []

    self.alice.train()
    self.bob.train()
    self.eve.train()

    # Training loop
    torch.autograd.set_detect_anomaly(True)

    # Hyperparams
    DECISION_MARGIN = 0.1
    STOP = False

    print("Starting training loop")
    print(f'Training with {BATCHES} batches over {EPOCHS} epochs')

    for E in range(EPOCHS):  
      print(f'Epoch {E + 1}/{EPOCHS}')

      for B in tqdm(range(BATCHES)):
        PLAIN = self.Plain.next(B)
        KEY = self.Key.next(B)
        K = torch.Tensor(KEY)

        for X in PLAIN:
          P0 = torch.Tensor(X[0])
          P1 = torch.Tensor(X[1])

          R = random.randint(0, 1)
          P = torch.Tensor(X[R])
          self.log('PLAIN', P)
                
          C = self.alice(torch.cat([P, K], dim=0))
          self.log('CIPHR', C)

          Pb = self.bob(torch.cat([C, K], dim=0))
          self.log('DCRPT', Pb)

          # Loss and BackProp
          bob_dec_loss = self.lossfn(Pb, P)
          Re = self.eve(torch.cat([P0, P1, C], dim=0))
          eve_adv_loss = self.lossfn(Re, torch.Tensor([1 - R, R]))
          bob_dec_loss = self.lossfn(Pb, P) + torch.square(1 - eve_adv_loss)
          
          opt_alice_bob.zero_grad()
          opt_eve.zero_grad()

          bob_dec_loss.backward(retain_graph=True)
          opt_alice_bob.step()

          if B > BATCHES/2:
            eve_adv_loss.backward(retain_graph=True)
            opt_eve.step()

          # torch.nn.utils.clip_grad_norm_(alice.parameters(), 4.0)
          # torch.nn.utils.clip_grad_norm_(bob.parameters(), 4.0)
          # torch.nn.utils.clip_grad_norm_(eve.parameters(), 4.0)

          bob_acc = 0
          for b in range(self.blocksize):
            if torch.abs(torch.round(Pb[b] - DECISION_MARGIN)) == P[b]:
              bob_acc += (1/self.blocksize)

          bob_bits_acc.append(bob_acc)
          bob_running_loss.append(bob_dec_loss.item())
          eve_running_loss.append(eve_adv_loss.item())

          if STOP:
            break

        if not self.debug:
          self.writer.add_scalar('Loss/Adversary', eve_adv_loss.item(), global_step=(E * BATCHES + B))
          self.writer.add_scalar('Loss/Training', bob_dec_loss.item(), global_step=(E * BATCHES + B))
          self.writer.add_scalar('Accuracy/Bits', torch.Tensor([bob_acc]), global_step=(E * BATCHES + B))
        
        if STOP:
          break
      
      if STOP:
        break

    self.log('Finished Training')
    self.writer.close()
    return (self.alice, self.bob, self.eve), (alice_running_loss, bob_running_loss, eve_running_loss)
