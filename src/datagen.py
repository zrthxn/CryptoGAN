import random

# Data and Proprocessing
BLOCKSIZE = 4
EPOCHS = 16
BATCHES = 256
BATCHLEN = 64

KEY = [random.randint(0, 1) for i in range(BLOCKSIZE)]
PLAIN = [[
    [random.randint(0, 1) for x in range(BLOCKSIZE)],
    [random.randint(0, 1) for y in range(BLOCKSIZE)]
    ] for i in range(BATCHLEN)]

def recalc_key():
  return [random.randint(0, 1) for i in range(BLOCKSIZE)]

def recalc_plain():
  return [[
    [random.randint(0, 1) for x in range(BLOCKSIZE)],
    [random.randint(0, 1) for y in range(BLOCKSIZE)]
  ] for i in range(BATCHLEN)]
  