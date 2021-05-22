import random
from src.util.gen import Generator

class KeyGenerator(Generator):
  silent = True
  
  def gen(self):
    return [
      [random.randint(0, 1) for _ in range(self.blocksize)]
    for _ in range(self.batchlen)] 


class PlainGenerator(Generator):
  silent = True
  
  def gen(self):
    return [
      [ [random.randint(0, 1) for _ in range(self.blocksize)], [random.randint(0, 1) for _ in range(self.blocksize)] ] 
    for _ in range(self.batchlen)] 

