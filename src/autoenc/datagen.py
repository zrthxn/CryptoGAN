import random
import numpy as np
import matplotlib.pyplot as plt


SAMPLERATE = 8 * np.pi
LEN = 250

def sine(f):
  x = np.arange(SAMPLERATE * 10)
  y = np.sin(2 * np.pi * f * x / SAMPLERATE)
  while len(y) != LEN:
    y = np.delete(y, len(y) - 1)
  
  return y


def dataset(BATCHLEN):
  BATCH = list()
  for _ in range(BATCHLEN):
    F1 = random.randint(1, 100)
    F2 = random.randint(1, 100)

    F0 = sine(F1) + sine(F2)
    BATCH.append([ F0, [F1, F2] ])

  return BATCH


def ghetto_tqdm(d, t, l = 10):
  done = int(l*(d/t))
  repeat = lambda c, i: ''.join([c for _ in range(i)])
  s = f'{d}/{t} [{repeat("=", done - 1)}>{repeat(".", l - (done - 1))}]'
  return s