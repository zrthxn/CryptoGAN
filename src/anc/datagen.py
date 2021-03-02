import random

class KeyGenerator():
  def __init__(self, BLOCKSIZE):
    self.blocksize = BLOCKSIZE
  
  def batchgen(self, BATCHES):
    return [[random.randint(0, 1) for _ in range(self.blocksize)] for _ in range(BATCHES)]

class PlainGenerator():
  def __init__(self, BLOCKSIZE, BATCHLEN):
    self.blocksize = BLOCKSIZE
    self.batchlen = BATCHLEN
  
  def batchgen(self, BATCHES):
    return [
      [[random.randint(0, 1) for _ in range(self.blocksize)] for _ in range(self.batchlen)] for _ in range(BATCHES)]
