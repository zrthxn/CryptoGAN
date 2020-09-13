import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.inlayer = nn.Linear(10, 20)

    def forward(self, *inputs):
        outputs = torch.Tensor()

        for _x in inputs:
            outputs.append(
                self.flow(_x)
            )
        
        return outputs

    def flow(self, _x):
        _x = self.inlayer(_x)
        _x = F.relu(_x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.inlayer = nn.Linear(20, 10)
    
    def forward(self, *inputs):
        pass
    