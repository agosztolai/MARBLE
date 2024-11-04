"""Layer module."""

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
import torch.nn.utils.parametrizations as param

from MARBLE import smoothing as s


class Diffusion(nn.Module):
    """Diffusion with learned t."""

    def __init__(self, tau0=0.0):
        """initialise."""
        super().__init__()

        self.diffusion_time = nn.Parameter(torch.tensor(float(tau0)))

    def forward(self, x, L, Lc=None, method="spectral"):
        """Forward."""
        if method == "spectral":
            assert len(L) == 2, "L must be a matrix or a pair of eigenvalues and eigenvectors"

        # making sure diffusion times are positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        t = self.diffusion_time

        if Lc is not None:
            out = s.vector_diffusion(x, t, Lc, L=L, method=method, normalise=True)
        else:
            out = [s.scalar_diffusion(x_, t, method, L) for x_ in x.T]
            out = torch.cat(out, axis=1)

        return out
    
    
class ParametrizedRotation(nn.Module):
    def __init__(self):
        super(ParametrizedRotation, self).__init__()
        # Define the single tunable parameter: the rotation angle (theta)
        self.theta = nn.Parameter(torch.tensor(0.0))  # Initial value of the angle

    def forward(self, x):
        # Construct the 2D rotation matrix using the tunable theta
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta,  cos_theta])
        ])
        
        # Apply the rotation matrix to the input
        rotated_x = torch.matmul(x, rotation_matrix)
        
        return rotated_x
    
    
class SOTransformation(nn.Module):
    def __init__(self, n):
        super(SOTransformation, self).__init__()
        # Initialize an identity matrix and use orthogonal parametrization
        self.matrix = nn.Parameter(torch.eye(n))

    def forward(self, x):
        # Perform QR decomposition to get an orthogonal matrix
        Q, R = torch.linalg.qr(self.matrix)

        # Ensure matrix is orthogonal by making diagonal of R positive
        diag_sign = torch.sign(torch.diag(R))
        Q = Q * diag_sign  # Adjust signs of Q

        # Check the determinant and flip one column if determinant is -1
        det = torch.det(Q)
        if det < 0:
            Q[:, 0] *= -1  # Flip the sign of the first column

        return torch.matmul(x, Q)
    
    
    
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
#         # On the Pubmed dataset, use `heads` output heads in `conv2`.
#         self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
#                              concat=False, dropout=0.6)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
    
class GCN(MessagePassing):
    def __init__(self):
        super(GCN, self).__init__(aggr='mean')  # 'mean' aggregation
    
    def forward(self, x, edge_index):
        # Compute normalization coefficients (optional)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j corresponds to the neighbors' features
        return x_j

    def update(self, aggr_out):
        # Return the aggregated features
        return aggr_out



class AnisoConv(MessagePassing):
    """Anisotropic Convolution"""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(aggr="add", **kwargs)

    def forward(self, x, kernels):
        """Forward pass."""
        out = []
        for K in kernels:
            out.append(self.propagate(K, x=x))

        # [[dx1/du, dx2/du], [dx1/dv, dx2/dv]] -> [dx1/du, dx1/dv, dx2/du, dx2/dv]
        out = torch.stack(out, axis=2)
        out = out.view(out.shape[0], -1)

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
    """Compute scaled inner-products between channel vectors.

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
