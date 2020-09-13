# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')
# %%
KEY = '1234'

PLAIN = [ [x/17 for x in range(0, 16)] for i in range(20) ]

# %%
def DenseBlock(shape):
  block = []
  for i in range(len(shape) - 1):
    block.append(nn.Linear(in_features=shape[i], out_features=shape[i + 1]))

  return block

# %%
class Sender(nn.Module):
  def __init__(self):
    super(Sender, self).__init__()

    self.in_block = nn.Sequential(
      *DenseBlock((16, 32, 32, 64, 16))
    )

  def forward(self, inputs):
    inputs = self.in_block(inputs)
    inputs = F.relu(inputs)

    inputs = F.sigmoid(inputs)
    return inputs


class Receiver(nn.Module):
  def __init__(self):
    super(Receiver, self).__init__()

    self.in_block = nn.Sequential(
      *DenseBlock((16, 32, 32, 64, 16))
    )

  def forward(self, inputs):
    inputs = self.in_block(inputs)
    inputs = F.relu(inputs)

    inputs = F.sigmoid(inputs)
    return inputs

# %%
alice = Sender()
bob = Receiver()
eve = Receiver()

lossfn = nn.BCELoss()
lossfn2 = nn.BCELoss()

opt = torch.optim.Adam(alice.parameters(), lr=0.1)
# %%

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(_):
  layers = [x for x in alice.parameters()]
  xloss = bob_reconst_loss
  ax1.clear()
  ax1.plot(xloss.detach())
  # for i in range(int(len(layers)/2)):
  #   ax1.plot(layers[i * 2].detach()[0], '.')

ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()

# from torch.utils.tensorboard import SummaryWriter

# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/experiment_1')


for P in PLAIN:
  P = torch.Tensor(P)

  cipher = alice(P).unsqueeze(dim=0)
  Pb = bob(cipher)
  Pe = eve(cipher)

  bob_reconst_loss = lossfn(Pb, P)
  eve_reconst_loss = lossfn(Pe, P)

  alice_loss = bob_reconst_loss - eve_reconst_loss

  alice_loss.backward(retain_graph=True)
  bob_reconst_loss.backward(retain_graph=True)
  eve_reconst_loss.backward(retain_graph=True)

  # ...log the running loss
  # writer.add_scalar('training loss',
  #   running_loss / 1000,
  #   epoch * len(trainloader) + i)

  # # ...log a Matplotlib Figure showing the model's predictions on a
  # # random mini-batch
  # writer.add_figure('predictions vs. actuals',
  #                 plot_classes_preds(alice, inputs, labels),
  #                 global_step=epoch * len(trainloader) + i)
  # running_loss = 0.0


  opt.step()

print('Finished Training') 
# %%
