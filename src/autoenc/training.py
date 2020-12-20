import itertools
from torch import Tensor
from torch import nn, optim

from model import Encoder
from datagen import dataset, sine, ghetto_tqdm

# enc = Encoder([250, 225, 200, 175, 125, 96, 60, 30, 10, 2])
enc = Encoder([250, 125, 2])
dec = Encoder([2, 125, 250])

# def dec(r):
#   r = r.tolist()
#   r = Tensor(sine(r[0]) + sine(r[1]))
#   r.requires_grad = False
#   return r

def test():
  pass

def train(DATA, EPOCHS):
  lfn = nn.L1Loss()
  prm = itertools.chain(enc.parameters(), dec.parameters())
  opt = optim.Adam(prm, lr=1e-5)

  avgloss = 0
  maxloss = 0
  for sample in DATA:
    X = Tensor(sample[0])
    # y = Tensor(sample[1])
    
    # for _ in range(EPOCHS):
    w = enc(X)
    z = dec(w)

    loss = lfn(X, z)
    loss.backward()
    opt.step()

    avgloss += loss.item()/(len(DATA))

    if loss.item() > maxloss:
      maxloss = loss.item()
  
  return avgloss, maxloss


# Init
if __name__ == "__main__":
  EPOCHS = 5
  BATCHES = 1024 * 2
  for B in range(BATCHES):
    data = dataset(64)
    avg, mxa = train(data, EPOCHS)

    print(f'{ghetto_tqdm(B, BATCHES)} : Avg {avg}, High {mxa}')
  