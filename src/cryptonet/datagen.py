import random

class KeyGenerator():
  def __init__(self, BLOCKSIZE):
    self.blocksize = BLOCKSIZE
  
  def single(self):
    return [random.randint(0, 1) for _ in range(self.blocksize)]

  def batchgen(self, BATCHES):
    return [ self.single() for _ in BATCHES ]

class PlainGenerator():
  def __init__(self, BLOCKSIZE, BATCHLEN):
    self.blocksize = BLOCKSIZE
    self.batchlen = BATCHLEN
  
  def single(self):
    return [random.randint(0, 1) for _ in range(self.blocksize)]

  def batchgen(self, BATCHES):
    return [
      [[self.single(), self.single()] for _ in range(self.batchlen)] 
    for _ in BATCHES]
