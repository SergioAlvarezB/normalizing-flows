import numpy as np
import torch
from torch import nn

from normalizing_flows.models.backbones import MLP


class Flow(nn.Module):
    def __init__(self, layers, **kwargs):
        super(Flow, self).__init__()

        self.layers = nn.ModuleList(layers)

        # A flow is (computationally) invertible if all its layers are.
        self.invertible = all([layer.invertible for layer in self.layers])

    def forward(self, x):
        cum_log_det = 0.0
        zs = []
        for layer in self.layers:
            x, log_det = layer(x)
            zs.append(x)
            cum_log_det += log_det

        return zs, cum_log_det

    def backward(self, z):
        if not self.invertible:
            raise ValueError('Flow inverse not tractable!')
        cum_log_det = 0.0
        xs = []
        for layer in self.layers[::-1]:
            z, log_det = layer.backward(z)
            xs.append(z)
            cum_log_det += log_det

        return xs, cum_log_det


class AffineConstantLayer(nn.Module):

    def __init__(self, dim, scale=True, shift=True):
        super(AffineConstantLayer, self).__init__()

        self.s = nn.Parameter(torch.zeros(1, dim, requires_grad=True)) \
            if scale else None
        self.t = nn.Parameter(torch.zeros(1, dim, requires_grad=True)) \
            if shift else None

        # Indicate wether flow is computationally invertible.
        self.invertible = True

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class NvpCouplingLayer(nn.Module):
    def __init__(self,
                 dim,
                 hidden_size=[5, 5],
                 scale=True,
                 shift=True,
                 random_flip=False):
        super(NvpCouplingLayer, self).__init__()
        self.s = MLP(dim, hidden_size, wscale=0.001) if scale \
            else lambda x: x.new_zeros(x.size())
        self.t = MLP(dim, hidden_size, wscale=0.001) if shift \
            else lambda x: x.new_zeros(x.size())

        mask = np.zeros((1, dim))
        mask[:, dim//2:] = 1

        self.mask = nn.Parameter(
                torch.as_tensor(mask.copy(), dtype=torch.float),
                requires_grad=False)

        # Indicate wether flow is computationally invertible.
        self.invertible = True
        self.random_flip = random_flip

        if random_flip:
            self.perm = np.random.permutation(dim)
            self.rev_perm = np.zeros(dim)
            self.rev_perm[self.perm] = np.arange(dim)
            self.perm = nn.Parameter(torch.LongTensor([self.perm]),
                                     requires_grad=False)
            self.rev_perm = nn.Parameter(torch.LongTensor([self.rev_perm]),
                                         requires_grad=False)

    def forward(self, x):
        x_b = self.mask*x
        b_1 = 1 - self.mask

        s, t = self.s(x_b), self.t(x_b)

        z = x_b + b_1 * (x * torch.exp(s) + t)

        log_det = torch.sum(b_1*s, dim=1).squeeze()
        if self.random_flip:
            z = z[:, self.perm.numpy().squeeze()]
        return z.flip((1,)), log_det

    def backward(self, z):
        z = z.flip((1,))
        if self.random_flip:
            z = z[:, self.rev_perm.numpy().squeeze()]
        x_b = self.mask*z
        b_1 = 1 - self.mask

        s, t = self.s(x_b), self.t(x_b)

        x = x_b + b_1*(z - t)*torch.exp(-s)

        log_det = torch.sum(b_1*(-s), dim=1).squeeze()
        return x, log_det


class PlanarLayer(nn.Module):
    def __init__(self, dim=0, params=None):
        super(PlanarLayer, self).__init__()

        # Initialize parameters.
        if params is not None:
            self.w = params['w'].squeeze()
            self.u = params['u'].squeeze()
            self.b = params['b'].squeeze()
        else:
            if dim < 1:
                raise ValueError('Either dim of params must be provided!')
            self.w = nn.Parameter(torch.rand(dim))
            self.u = nn.Parameter(torch.rand(dim))
            self.b = nn.Parameter(torch.rand(1))

        # Indicate wether flow is computationally invertible.
        self.invertible = False

    def forward(self, x):
        # Ensure w*u >= 1 to mantain invertibility.
        wtu = torch.dot(self.w, self.u)
        m = -1 + torch.log1p(torch.exp(wtu))

        u_hat = self.u + (m - wtu)*self.w/torch.norm(self.w)

        # Forward pass
        h = torch.tanh(torch.matmul(x, self.w) + self.b)
        z = x + torch.matmul(h.view(-1, 1), u_hat.view(1, -1))

        # Log-determinant
        hp = 1 - h**2
        phi = torch.matmul(hp.view(-1, 1), self.w.view(1, -1))
        det = torch.abs(1 + torch.matmul(phi, u_hat.view(-1, 1)))
        log_det = torch.log(det.squeeze())

        return z, log_det


class RadialLayer(nn.Module):
    def __init__(self, dim):
        super(RadialLayer, self).__init__()

        # Initialize parameters
        self.z0 = nn.Parameter(torch.rand(dim))
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        # Indicate wether flow is computationally invertible.
        self.invertible = False

    def forward(self, x):
        # Parametrize b to ensure invertibility.
        b_hat = -self.a + torch.log1p(torch.exp(self.b))

        x_z0 = x - self.z0
        h = 1./(self.a + torch.norm(x_z0, dim=1, keepdim=True))

        z = x + b_hat*h.expand_as(x_z0)*x_z0

        # Log-determinant
        det = log_det = torch.tensor(1.0)
        log_det = torch.log(det)

        return z, log_det
