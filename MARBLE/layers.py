"""Layer module."""
import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

from MARBLE import smoothing as s



class OrthogonalTransformLayer(nn.Module):
    """ Learn orthogonal transformation of output embeddings """
    
    def __init__(self,
                 size,
                 initial_rotation=None,
                 enforce_rotation=True,
                 enforce_orthogonality=True
                 ):
        super(OrthogonalTransformLayer, self).__init__()
        self.size = size
        self.enforce_rotation = enforce_rotation
        self.enforce_orthogonality = enforce_orthogonality
        # Initialize the identity matrix 
        
        if initial_rotation is None:
            Q = torch.eye(size) # torch.eye(size)# torch.eye(size)#  torch.randn(size, size)
        else:
            Q = initial_rotation # assign an initial rotation matrix
        #Q = torch.eye(size)
        
        # Use QR decomposition to orthogonalize it
        #Q, _ = torch.linalg.qr(Q)
        # Ensure the determinant is +1 for a proper rotation
        if self.enforce_rotation:
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]  # Flip the sign of the first column if the determinant is negative
        # Wrap as a parameter
        self.Q = nn.Parameter(Q)

    def forward(self, x):
        
        if self.enforce_orthogonality:
            self.polar_decomp()
            
        if self.enforce_rotation:
            self.check_determinant()

        return torch.matmul(x, self.Q.T) 

    def check_determinant(self):

        if self.enforce_rotation:
            if torch.det(self.Q) < 0:
                self.Q[:, 0] = -self.Q[:, 0]  # Flip the sign of the first column if the determinant is negative
        
    def polar_decomp(self):   # express polar decomposition in terms of singular-value decomposition
        U, S, Vh = torch.linalg.svd(self.Q)
        u = U @ Vh
        #p = Vh.T.conj() @ S.diag().to(dtype = self.Q.dtype) @ Vh
        self.Q.data = u
        #return  u, p



class AffineTransformLayer(nn.Module):
    """ Learn affine transformation of output embeddings """
    
    def __init__(self, size):
        super(AffineTransformLayer, self).__init__()
        self.size = size
        
        # Initialize the linear transformation matrix
        # It doesn't need to be orthogonal, so we can initialize it normally
        self.A = nn.Parameter(torch.eye(size))#nn.Parameter(torch.randn(size, size))
        
        # Initialize the translation vector
        self.b = nn.Parameter(torch.ones(size)) #nn.Parameter(torch.randn(size))
        
    def forward(self, x):
        # Apply the affine transformation: Ax + b
        return torch.matmul(x, self.A) + self.b
    

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


class AnisoConv(MessagePassing):
    """Anisotropic Convolution"""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(aggr="add", **kwargs)

    def forward(self, x, kernels):
        """Forward."""
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
            #dim_ = int(K_t.size(dim=1) / n) # added this 

        x = K_t.matmul(x, reduce=self.aggr)
        
        #if x.shape[0]==
        #dim_ = int(x.shape[0]/K_t.size(0))
        #return x.view(-1, dim_)
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
