import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 dim,
                 hidden_size=[],
                 activation=F.relu,
                 wscale=1.):
        super(MLP, self).__init__()
        self.activation = activation
        units = [dim] + hidden_size + [dim]

        layers = []
        for inp, outp in zip(units[:-1], units[1:]):
            linear = nn.Linear(inp, outp)
            with torch.no_grad():
                linear.weight = nn.Parameter(linear.weight.detach()*wscale)
                linear.bias = nn.Parameter(linear.bias.detach()*wscale)
            layers.append(linear)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        y = self.layers[-1](x)

        return y


class TempScaler(nn.Module):
    def __init__(self):
        super(TempScaler, self).__init__()

        self.T = nn.Parameter(torch.ones(1))

    def forward(self, x):
        z = x/torch.abs(self.T)

        return z

    def backward(self, z):
        x = z*torch.abs(self.T)

        return x
