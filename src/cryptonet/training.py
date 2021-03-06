import itertools
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from cryptonet.datagen import KeyGenerator as Key
from cryptonet.datagen import PlainGenerator as Plain

from cryptonet.model import KeyholderNetwork, AttackerNetwork
from cryptonet.model import weights_init_normal

from autoenc.datagen import ghetto_tqdm

BLOCKSIZE = 4
VERSION = 1.1

class TrainingSession():
  def __init__(self, debug = False):
    # Initialize Networks
    self.alice = KeyholderNetwork(BLOCKSIZE)
    self.bob = KeyholderNetwork(BLOCKSIZE)
    self.eve = AttackerNetwork(BLOCKSIZE)

    # Initialize weights
    # self.alice.apply(weights_init_normal)
    # self.bob.apply(weights_init_normal)
    # self.eve.apply(weights_init_normal)

    # # CUDA
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # # torch.set_default_tensor_type('torch.cuda.Tensor' if torch.cuda.is_available() else 'torch.Tensor')
    # print('Using device', device)

    # self.alice.to(device)
    # self.bob.to(device)
    # self.eve.to(device)

    # self.lossfn = torch.nn.L1Loss()
    self.lossfn = torch.nn.MSELoss()
    # self.lossfn = torch.nn.CrossEntropyLoss()
    # self.lossfn = torch.nn.BCELoss()

    self.debug = debug
    self.writer = SummaryWriter(f'training/cryptonet_vL{VERSION}') if not debug else None

  def log(self, *ip):
    if self.debug:
      print(*ip)

  def train(self, BATCHLEN = 64, BATCHES = 256, EPOCHS = 16):
    KeyGenerator = Key(BLOCKSIZE)
    PlainGenerator = Plain(BLOCKSIZE, BATCHLEN)

    ab_params = itertools.chain(self.alice.parameters(), self.bob.parameters())
    opt_alice_bob = torch.optim.Adam(ab_params, lr=0.0008, weight_decay=1e-5)
    opt_eve = torch.optim.Adam(self.eve.parameters(), lr=0.001)

    if not self.debug:
      graph_ip = torch.cat([torch.Tensor(PlainGenerator.single()), torch.Tensor(KeyGenerator.single())], dim=0)
      self.writer.add_graph(self.alice, graph_ip)
      self.writer.close()

    alice_running_loss = []
    bob_running_loss = []
    eve_running_loss = []

    bob_bits_acc = []

    print(f'Cryptonet Model v{VERSION}')
    print(f'Training with {BATCHES * BATCHLEN} samples over {EPOCHS} epochs')

    self.alice.train()
    self.bob.train()
    self.eve.train()

    # Training loop
    torch.autograd.set_detect_anomaly(True)

    # Hyperparams
    DECISION_MARGIN = 0.1
    STOP = False

    for E in range(EPOCHS):  
      print(f'Epoch {E + 1}/{EPOCHS}')
      
      KEY = KeyGenerator.batch()
      PLAIN = PlainGenerator.batch()
      K = torch.Tensor(KEY)

      for B in range(BATCHES):
        # print(ghetto_tqdm(B, BATCHES))

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
          Re = self.eve(torch.cat([P0, P1, C.detach()], dim=0))
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
          for b in range(BLOCKSIZE):
            if torch.abs(torch.round(Pb[b] - DECISION_MARGIN)) == P[b]:
              bob_acc += (1/BLOCKSIZE)

          bob_bits_acc.append(bob_acc)
          bob_running_loss.append(bob_dec_loss.item())
          eve_running_loss.append(eve_adv_loss.item())

          if STOP:
            break

        if not self.debug:
          self.writer.add_scalar('Training Loss', bob_dec_loss.item(), (E * BATCHES) + B)
          self.writer.add_scalar('Bit Accuracy', torch.Tensor([bob_acc]), (E * BATCHES) + B)
          self.writer.add_scalar('Adversary Loss', eve_adv_loss.item(), (E * BATCHES)  + B)
          self.writer.close()
        
        if STOP:
          break
      
      if STOP:
        break

    self.log('Finished Training')
    return (self.alice, self.bob, self.eve), (alice_running_loss, bob_running_loss, eve_running_loss)
