import numpy as np
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt

# Evaluation Plots
sns.set_style('whitegrid')
sf = 5000 #min(int(BATCHES * EPOCHS/10), 50)

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
  
def plot(losses):
  # plt.plot(trendline(alice_running_loss[:1000], sf))
  plt.plot(bob_running_loss)
  plt.plot(trendline(bob_running_loss, sf), color='black')
  # plt.plot(trendline(eve_running_loss, sf))
  plt.legend(['Bob', 'Eve'], loc='upper right')
  # plt.legend(['Alice', 'Bob', 'Eve'], loc='upper right')
  # plt.xlim(len(alice_running_loss) - 1000, len(alice_running_loss))
  plt.xlabel('Samples')
  plt.ylabel(f'Loss (SF {sf})')
  plt.title(f'Training Loss')
  if SAVEPLOT:
    plt.savefig(f'../models/cryptonet/graphs/loss_{datetime.now()}.png', dpi=400)
  plt.show()

  plt.plot(bob_bits_acc, color='red')
  plt.plot(trendline(bob_bits_acc, sf), color='black')
  plt.legend(['Actual', 'Trend'], loc='upper right')
  plt.xlabel('Samples')
  plt.ylabel(f'Bit error (SF {sf})')
  plt.title(f'Bit Error')
  if SAVEPLOT:
    plt.savefig(f'../models/cryptonet/graphs/error_{datetime.now()}.png', dpi=400)
  plt.show()
