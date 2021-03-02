import itertools
from torch import Tensor
from torch import nn, optim

from autoenc.model import Encoder
from autoenc.datagen import dataset, sine, ghetto_tqdm

class TrainingSession():
  def __init__(self):
    # enc = Encoder([250, 225, 200, 175, 125, 96, 60, 30, 10, 2])
    self.enc = Encoder([250, 125, 2])
    self.dec = Encoder([2, 125, 250])

    self.lossfn = nn.L1Loss()

  def test(self):
    pass

  def train(self, BATCHES, EPOCHS):
    prm = itertools.chain(self.enc.parameters(), self.dec.parameters())
    opt = optim.Adam(prm, lr=1e-3)

    avgloss = 0
    maxloss = 0

    data = dataset(BATCHES)

    for E in EPOCHS:
      for B in range(BATCHES):
        X = Tensor(data[B])
        # y = Tensor(sample[1])
        
        # for _ in range(EPOCHS):
        w = self.enc(X)
        z = self.dec(w)

        loss = self.lossfn(X, z)
        loss.backward()
        opt.step()

        avgloss += loss.item()/len(data)

        if loss.item() > maxloss:
          maxloss = loss.item()
      
    print(f'{ghetto_tqdm(B, BATCHES)} : Avg {avgloss}, High {maxloss}')
    return avgloss, maxloss
