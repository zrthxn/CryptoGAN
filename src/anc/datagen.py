import random
from tqdm import tqdm

VALUES = [-1, 1]

class KeyGenerator():
  def __init__(self, BLOCKSIZE):
    self.blocksize = BLOCKSIZE
  
  def batchgen(self, BATCHES):
    print(f'Generating keys...')
    return [
      [ VALUES[random.randint(0, 1)] for _ in range(self.blocksize)] 
    for _ in tqdm(range(BATCHES))]

class PlainGenerator():
  def __init__(self, BLOCKSIZE, BATCHLEN):
    self.blocksize = BLOCKSIZE
    self.batchlen = BATCHLEN
  
  def batchgen(self, BATCHES):
    print(f'Generating plaintexts...')
    return [
      [[ VALUES[random.randint(0, 1)] for _ in range(self.blocksize)] for _ in range(self.batchlen)] 
    for _ in tqdm(range(BATCHES))]
