"""Layer module."""
import torch
from torch import nn
from torch.nn.functional import normalize
from torch_geometric.nn.conv import MessagePassing

from MARBLE import geometry as g

class SkipMLP(nn.Module):
    """ MLP with skip connections """

    def __init__(self, channel_list, dropout=0.0, bias=True):
        super(SkipMLP, self).__init__()
        assert len(channel_list) > 1, "Channel list must have at least two elements for an MLP."
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.in_channels = channel_list[0]
        for i in range(len(channel_list) - 1):
            self.layers.append(nn.Linear(channel_list[i], channel_list[i+1], bias=bias))
            if i < len(channel_list) - 2:  # Don't add activation or dropout to the last layer
                self.layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        identity = x
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if x.shape[1] == layer.weight.shape[1]:  # Check if skip connection is possible
                    identity = x  # Save identity for skip connection
                x = layer(x)
                if x.shape[1] == identity.shape[1]:  # Apply skip connection if shapes match
                    x += identity
            else:
                x = layer(x)  # Apply activation or dropout
        return x


class Diffusion(nn.Module):
    """Diffusion with learned t."""

    def __init__(self, tau0=0.0):
        """initialise."""
        super().__init__()

        self.diffusion_time = nn.Parameter(torch.tensor(float(tau0)))

    def forward(self, x, L, Lc=None, method="spectral"):
        """Forward."""
        if method == "spectral":
            assert (
                len(L) == 2
            ), "L must be a matrix or a pair of eigenvalues \
                                and eigenvectors"

        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        t = self.diffusion_time

        if Lc is not None:
            out = g.vector_diffusion(x, t, method, Lc)
        else:
            out = [g.scalar_diffusion(x_, t, method, L) for x_ in x.T]
            out = torch.cat(out, axis=1)

        return out


class AnisoConv(MessagePassing):
    """Anisotropic Convolution"""

    def __init__(self, vec_norm=False, **kwargs):
        """Initialize."""
        super().__init__(aggr="add", **kwargs)

        self.vec_norm = vec_norm

    def forward(self, x, kernels):
        """Forward."""
        out = []
        for K in kernels:
            out.append(self.propagate(K, x=x))

        # [[dx1/du, dx2/du], [dx1/dv, dx2/dv]] -> [dx1/du, dx1/dv, dx2/du, dx2/dv]
        out = torch.stack(out, axis=2)
        out = out.view(out.shape[0], -1)

        # if self.vec_norm:
        #     out = normalize(out, dim=-1, p=2)

        return out

    def message_and_aggregate(self, K_t, x):
        """Message passing step. If K_t is a txs matrix (s sources, t targets),
        do matrix multiplication K_t@x, broadcasting over column features.
        If K_t is a t*dimxs*dim matrix, in case of manifold computations,
        then first reshape, assuming that the columns of x are ordered as
        [dx1/du, x1/dv, ..., dx2/du, dx2/dv, ...].
        """
        n, dim = x.shape

        if (K_t.size(dim=1) % n * dim) == 0:
            n_ch = torch.div(n * dim, K_t.size(dim=1), rounding_mode="floor")
            x = x.view(-1, n_ch)

        x = K_t.matmul(x, reduce=self.aggr)

        return x.view(-1, dim)


class InnerProductFeatures(nn.Module):
    r"""Compute scaled inner-products between channel vectors.

    Input: (V x C*D) vector of (V x n_i) list of vectors with \sum_in_i = C*D
    Output: (VxC) dot products
    """

    def __init__(self, C, D):
        super().__init__()

        self.C, self.D = C, D

        self.O_mat = nn.ModuleList()
        for _ in range(C):
            self.O_mat.append(nn.Linear(D, D, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        for i, _ in enumerate(self.O_mat):
            self.O_mat[i].weight.data = torch.eye(self.D)

    def forward(self, x):
        """Forward."""
        if not isinstance(x, list):
            x = [x]

        x = [x_.view(x_.shape[0], -1, self.D) for x_ in x]

        # for scalar signals take magnitude
        if self.D == 1:
            x = [x_.norm(dim=2) for x_ in x]

            return torch.cat(x, axis=1)

        # for vector signals take inner products
        # bring to form where all columns are vector in the tangent space
        # so taking inner products is possible
        # [ x1 dx1/du ...]
        #  x2 dx1/dv
        #  x3 dx1/dw
        x = [x_.swapaxes(1, 2) for x_ in x]
        x = torch.cat(x, axis=2)

        assert x.shape[2] == self.C, "Number of channels is incorrect!"

        # O_ij@x_j
        Ox = [self.O_mat[j](x[..., j]) for j in range(self.C)]
        Ox = torch.stack(Ox, dim=2)

        # \sum_j x_i^T@O_ij@x_j
        xOx = torch.einsum("bki,bkj->bi", x, Ox)

        return torch.tanh(xOx).reshape(x.shape[0], -1)
