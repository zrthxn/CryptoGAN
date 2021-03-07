import itertools
from torch import Tensor
from torch import nn, optim
from tqdm import tqdm

from autoenc.model import Encoder, Decoder
from autoenc.datagen import dataset, sine, ghetto_tqdm

class TrainingSession():
  def __init__(self):
    # enc = Encoder([250, 225, 200, 175, 125, 96, 60, 30, 10, 2])
    self.enc = Encoder()
    self.dec = Decoder()
    self.eve = Decoder()

    self.lossfn = nn.L1Loss()

  def train(self, BATCHES, EPOCHS):
    prm = itertools.chain(self.enc.parameters(), self.dec.parameters())
    opt = optim.Adam(prm, lr=1e-3)

    data = dataset(BATCHES)

    for E in EPOCHS:
      for B in tqdm(range(BATCHES)):
        P = Tensor([1,1,1,1,1,1,1,1])
        K = Tensor([1,1,1,1,1,1,1,1])

        X = P.matmul(K.T())

        z = self.enc(X)
        
        x = self.dec(z)
        w = self.eve(z)
        
        p = x.matmul(K)
        loss = self.lossfn(p, P)

        loss.backward()
        opt.step()
      
    return self.enc, self.dec
