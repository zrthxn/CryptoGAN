import random

class KeyGenerator():
  def __init__(self, BLOCKSIZE):
    self.blocksize = BLOCKSIZE
  
  def batchgen(self):
    return [random.randint(0, 1) for i in range(self.blocksize)]

class PlainGenerator():
  def __init__(self, BLOCKSIZE, BATCHLEN):
    self.blocksize = BLOCKSIZE
    self.batchlen = BATCHLEN
  
  def batchgen(self):
    return [[
      [random.randint(0, 1) for x in range(self.blocksize)],
      [random.randint(0, 1) for y in range(self.blocksize)]
    ] for i in range(self.batchlen)]
