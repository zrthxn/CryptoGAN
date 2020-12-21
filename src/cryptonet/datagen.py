import random

class KeyGenerator():
  def __init__(self, BLOCKSIZE):
    self.blocksize = BLOCKSIZE
  
  def single(self):
    return [random.randint(0, 1) for i in range(self.blocksize)]

  def batch(self):
    return [random.randint(0, 1) for i in range(self.blocksize)]

class PlainGenerator():
  def __init__(self, BLOCKSIZE, BATCHLEN):
    self.blocksize = BLOCKSIZE
    self.batchlen = BATCHLEN
  
  def single(self):
    return [random.randint(0, 1) for y in range(self.blocksize)]

  def batch(self):
    return [[self.single(), self.single()] for i in range(self.batchlen)]
