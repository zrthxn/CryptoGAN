import random
from src.util.gen import Generator

VALUES = [-1, 1]

class KeyGenerator(Generator):
  silent = True
  
  def gen(self):
    return [
      [ VALUES[random.randint(0, 1)] for _ in range(self.blocksize) ] 
    for _ in range(self.batchlen)]


class PlainGenerator(Generator):
  silent = True

  def gen(self):
    return [
      [ VALUES[random.randint(0, 1)] for _ in range(self.blocksize)] 
    for _ in range(self.batchlen)]