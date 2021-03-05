import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Evaluation Plots
sns.set_style('whitegrid')
sf = 5000 #min(int(BATCHES * EPOCHS/10), 50)

TITLE_TAG = f'{BLOCKSIZE} bits, {dist}, B={BETA} G={GAMMA} W={OMEGA}'
FILE_TAG = f'{EPOCHS}E{BATCHES}x{BATCHLEN}v{VERSION}'

SAVEPLOT = False
# Turn this line on and off to control plot saving
# SAVEPLOT = True

def trendline(data, deg=1):
  for _ in range(deg):
    last = data[0]
    trend = []
    for x in data:
      trend.append((x+last)/2)
      last = x
    
    data = trend
 
  return trend


# def ghetto_tqdm(d, t, l = 10):
#   done = int(l*(d/t))
#   repeat = lambda c, i: ''.join([c for _ in range(i)])
#   print(f'{d}/{t} [{repeat("=", done - 1)}>{repeat(".", l - (done - 1))}]')
  

# plt.plot(trendline(alice_running_loss[:1000], sf))
plt.plot(bob_running_loss)
plt.plot(trendline(bob_running_loss, sf), color='black')
# plt.plot(trendline(eve_running_loss, sf))
plt.legend(['Bob', 'Eve'], loc='upper right')
# plt.legend(['Alice', 'Bob', 'Eve'], loc='upper right')
# plt.xlim(len(alice_running_loss) - 1000, len(alice_running_loss))
plt.xlabel('Samples')
plt.ylabel(f'Loss (SF {sf})')
plt.title(f'Training Loss - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/loss_{FILE_TAG}.png', dpi=400)
plt.show()

plt.plot(bob_bits_acc, color='red')
plt.plot(trendline(bob_bits_acc, sf), color='black')
plt.legend(['Actual', 'Trend'], loc='upper right')
plt.xlabel('Samples')
plt.ylabel(f'Bit error (SF {sf})')
plt.title(f'Bit Error - {TITLE_TAG}')
if SAVEPLOT:
  plt.savefig(f'../models/cryptonet/graphs/error_{FILE_TAG}.png', dpi=400)
plt.show()
