import random
from src.util.gen import Generator

class KeyGenerator(Generator):
  silent = True
  
  def gen(self):
    return [float(random.randint(0, 1)) for _ in range(self.blocksize)]


class PlainGenerator(Generator):
  silent = True
  
  def gen(self):
    return [float(random.randint(0, 1)) for _ in range(self.blocksize)]
