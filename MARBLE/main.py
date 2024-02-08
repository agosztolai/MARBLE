"""Main network"""
import glob
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as opt
import yaml
from torch import nn
from torch_geometric.nn import MLP
from tqdm import tqdm

from functools import partial

from procrustes.generalized import generalized
from typing import List, Tuple, Optional

from MARBLE import dataloader
from MARBLE import geometry
from MARBLE import layers
from MARBLE import utils


class net(nn.Module):
    """MARBLE neural network.

    The possible parameters and their default values are described below,
    and can be accessed via the `params` dictionnary in this class constructor.

    Args:
        batch_size: batch size (default=64)
        epochs: optimisation epochs (default=20)
        lr: iniital learning rate (default=0.01)
        momentum: momentum (default=0.9)
        diffusion: set to True to use diffusion layer before gradient computation (default=False)
        include_positions: include positions as features (warning: this is untested) (default=False)
        include_self: include vector at the center of feature (default=True)
        order: order to which to compute the directional derivatives (default=2)
        inner_product_features: transform gradient features to inner product features (default=True)
        frac_sampled_nb: fraction of neighbours to sample for gradient computation
            (if -1 then all neighbours) (default=-1)
        dropout: dropout in the MLP (default=0.)
        hidden_channels: number of hidden channels (default=16). If list, then adds multiple layers.
        out_channels: number of output channels (if null, then =hidden_channels) (default=3)
        bias: learn bias parameters in MLP (default=True)
        vec_norm: normalise features at each derivative order to unit length (default=False)
        emb_norm: normalise MLP output to unit length (default=False)
        batch_norm: batch normalisation (default=True)
        seed: seed for reproducibility
    """

    def __init__(self, data, loadpath=None, params=None, verbose=True):
        """
        Constructor of the MARBLE net.

        Args:
            data: PyG data
            loadpath: path to a model file, or a directory with models (best model will be used)
            params: dict with parameters to overwrite default params or a path to a yaml file
            verbose: run in verbose mode
        """
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if loadpath is not None:
            if Path(loadpath).is_dir():
                loadpath = max(glob.glob(f"{loadpath}/best_model*"))
            self.params = torch.load(loadpath, map_location=device)["params"]
        else:
            if params is not None:
                self.params = params
            else:
                self.params = {}

        self._epoch = 0  # to resume optimisation
        self.parse_parameters(data)
        self.check_parameters(data)
        self.setup_layers()
        self.loss = loss_fun()
        self.loss_orth = ortho_loss()
        self.reset_parameters()

        if verbose:
            utils.print_settings(self)

        if loadpath is not None:
            self.load_model(loadpath)

    def parse_parameters(self, data):
        """Load default parameters and merge with user specified parameters"""

        file = os.path.dirname(__file__) + "/default_params.yaml"
        with open(file, "rb") as f:
            params = yaml.safe_load(f)

        params["dim_signal"] = data.x.shape[1]
        params["dim_emb"] = data.pos.shape[1]
        params["n_graphs"] = data.num_graphs

        if hasattr(data, "dim_man"):
            params["dim_man"] = data.dim_man

        # merge dictionaries without duplications
        for key in params.keys():
            if key not in self.params.keys():
                self.params[key] = params[key]

        if params["frac_sampled_nb"] != -1:
            self.params["n_sampled_nb"] = int(data.degree * params["frac_sampled_nb"])
        else:
            self.params["n_sampled_nb"] = -1

        if self.params["batch_norm"]:
            self.params["batch_norm"] = "batch_norm"
        else:
            self.params["batch_norm"] = None

    def check_parameters(self, data):
        """Check parameter validity"""

        assert self.params["order"] > 0, "Derivative order must be at least 1!"

        if self.params["vec_norm"]:
            assert data.x.shape[1] > 1, "Using vec_norm=True is not permitted for scalar signals"

        if self.params["diffusion"]:
            assert hasattr(data, "L"), "No Laplacian found. Compute it in preprocessing()!"

        pars = [
            "batch_size",
            "epochs",
            "lr",
            "momentum",
            "order",
            "inner_product_features",
            "dim_signal",
            "dim_emb",
            "frac_sampled_nb",
            "dropout",
            "diffusion",
            "hidden_channels",
            "out_channels",
            "bias",
            "batch_norm",
            "vec_norm",
            "emb_norm",
            "global_align",
            "seed",
            "include_positions",
            "include_self",
        ]

        for p in pars:
            assert p in list(self.params.keys()), f"Parameter {p} is not specified!"

    def reset_parameters(self):
        """reset parmaeters."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def setup_layers(self):
        """Setup layers."""

        s, d, o = self.params["dim_signal"], self.params["dim_emb"], self.params["order"]
        if "dim_man" in self.params.keys():
            if self.params["global_align"]:
                d = self.params["dim_man"]
            else:
                s = d = self.params["dim_man"]

        # diffusion
        self.diffusion = layers.Diffusion()

        # gradient features
        self.grad = nn.ModuleList(layers.AnisoConv() for i in range(o))

        # cumulated number of channels after gradient features
        cum_channels = s * (1 - d ** (o + 1)) // (1 - d)
        if self.params["inner_product_features"]:
            cum_channels //= s
            if s == 1:
                cum_channels = o + 1

            self.inner_products = layers.InnerProductFeatures(cum_channels, s)
        else:
            self.inner_products = None

        if self.params["include_positions"]:
            # cum_channels += d
            cum_channels += self.params["dim_signal"]
            
        if not self.params["include_self"]:
            cum_channels -= s

        # encoder
        if not isinstance(self.params["hidden_channels"], list):
            self.params["hidden_channels"] = [self.params["hidden_channels"]]

        channel_list = (
            [cum_channels] + self.params["hidden_channels"] + [self.params["out_channels"]]
        )

        self.enc = MLP(
            channel_list=channel_list,
            dropout=self.params["dropout"],
            bias=self.params["bias"],
            norm=self.params["batch_norm"],
        )
        
        self.orthogonal_transform = nn.ModuleList([
                                        layers.OrthogonalTransformLayer(self.params["out_channels"]) for _ in range(self.params["n_graphs"])
                                    ])
        
        self.affine_transform = nn.ModuleList([
                                        layers.AffineTransformLayer(self.params["out_channels"]) for _ in range(self.params["n_graphs"])
                                    ])        

    def forward(self, data, n_id, adjs=None, n_batch=None):
        """Forward pass.
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to
        simplify message passing. By convention, the first size[1] entries of x
        are the target nodes, i.e, x = concat[x_target, x_other]."""

        x = data.x
        n, d = x.shape[0], data.gauges.shape[2]
        dim_man = data.gauges.shape[2]
        mask = data.mask

        # diffusion
        if self.params["diffusion"]:
            if hasattr(data, "Lc"):
                x = geometry.global_to_local_frame(x, data.gauges)
                x = self.diffusion(x, data.L, Lc=data.Lc, method="spectral")
                x = geometry.global_to_local_frame(x, data.gauges, reverse=True)
            else:
                x = self.diffusion(x, data.L, method="spectral")

        # local gauges
        if self.params["inner_product_features"] or self.params['global_align']:
            x = geometry.global_to_local_frame(x, data.gauges)

        # restrict to current batch
        x = x[n_id]
        mask = mask[n_id]
        n_id_orig = n_id
        if data.kernels[0].size(0) == n * d:
            n_id = utils.expand_index(n_id, d)
        else:
            d = 1

        if self.params["vec_norm"]:
            x = F.normalize(x, dim=-1, p=2)

        # gradients
        if self.params["include_self"]:
            out = [x]
        else:
            out = []
            
        for i, (_, _, size) in enumerate(adjs):
            kernels = [K[n_id[: size[1] * d], :][:, n_id[: size[0] * d]] for K in data.kernels]

            x = self.grad[i](x, kernels)

            if self.params["vec_norm"]:
                x = F.normalize(x, dim=-1, p=2)

            out.append(x)

        last_size = adjs[-1][2]
        # take target nodes
        out = [o[: last_size[1]] for o in out]
            
        if self.params["global_align"]:            
            
            for i, o in enumerate(out):
                # project back into ambient coordinate space
                # for each output only act on each local tangent direction e.g. [dx/du, dx/dv]
                new_o = [geometry.global_to_local_frame(o[:,d_*dim_man:d_*dim_man+dim_man], data.gauges[n_id_orig][:last_size[1]], reverse=True)
                         for d_ in range(int(o.shape[1]/dim_man))]
                new_o = torch.cat(new_o, axis=1)
                out[i] = new_o
                            

        # inner products
        if self.params["inner_product_features"]:
            out = self.inner_products(out)
        else:
            out = torch.cat(out, axis=1)

        if self.params["include_positions"]:
            out = torch.hstack([data.pos[n_id_orig[: last_size[1]]], out])

        emb = self.enc(out)      
        
        if self.params["emb_norm"]:  # spherical output
            emb = F.normalize(emb)   

        # learn orthogonal transformation
        if self.params["global_align"]:                
            emb_, indices = group_embeddings_by_graph(n_id_orig[:n_batch], data.batch, emb, limit_rows=False)
            #ortho = [self.orthogonal_transform[i](emb_[i]) for i in range(len(emb_))]
            ortho = [self.orthogonal_transform[i](emb_[i]) for i in range(len(emb_))]
    
            emb = torch.zeros_like(emb)
            #ortho = torch.cat(ortho, dim=0)
            #indices = torch.cat(indices, dim=0).squeeze()
            emb[torch.cat(indices, dim=0).squeeze()] = torch.cat(ortho, dim=0)    

        # if self.params["emb_norm"]:  # spherical output
        #     emb = F.normalize(emb)   

        return emb, mask[: last_size[1]]

    def evaluate(self, data):
        """Evaluate."""
        warnings.warn("MARBLE.evaluate() is deprecated. Use MARBLE.transform() instead.")
        return self.transform(data)

    def transform(self, data):
        """Forward pass @ evaluation (no minibatches)"""
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = utils.EdgeIndex(data.edge_index, torch.arange(data.edge_index.shape[1]), size)
            adjs = utils.to_list(adjs) * self.params["order"]

            try:
                data.kernels = [
                    utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()).t()
                    for K in utils.to_list(data.kernels)
                ]
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            _, data, adjs = utils.move_to_gpu(self, data, adjs)
            out, _ = self.forward(data, torch.arange(len(data.x)), adjs)
            utils.detach_from_gpu(self, data, adjs)

            data.emb = out.detach().cpu()

            return data

    def batch_loss(self, data, loader, train=False, verbose=False, optimizer=None):
        """Loop over minibatches provided by loader function.

        Args:
            x : (nxdim) feature matrix
            loader : dataloader object from dataloader.py

        """

        if train:  # training mode (enables dropout in MLP)
            self.train()

        if verbose:
            print("\n")

        cum_loss = 0
        for batch in tqdm(loader, disable=not verbose):
            n_batch, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]

            emb, mask = self.forward(data, n_id, adjs, n_batch)
            loss = self.loss(emb, mask, data, n_id, n_batch)
            cum_loss += float(loss)
            
            # computing loss on orthogonal transformations
            #custom_loss = self.loss_orth(emb, data, n_id, n_batch)
            #cum_loss += float(custom_loss.mean())

            if optimizer is not None:
                optimizer.zero_grad()  # zero gradients, otherwise accumulates
                loss.backward(retain_graph=True)  # backprop

                # looping over and back propgating only on the orthogonal transform layers
                #for i,layer in enumerate(self.orthogonal_transform):
                #    layer.Q.grad = None
                #    custom_loss[i].backward(retain_graph=True)

                #self.orthogonal_transform[0].Q.grad = None
                
                optimizer.step()

        self.eval()

        return cum_loss / len(loader), optimizer

    def run_training(self, data, outdir=None, verbose=False):
        """Run training."""
        warnings.warn("MARBLE.run_training() is deprecated. Use MARBLE.fit() instead.")

        self.fit(data, outdir=outdir, verbose=verbose)

    def fit(self, data, outdir=None, verbose=False):
        """Network training.

        Args:
            data: PyG data
            outdir: folder to save intermediate models
            verbose: run in verbose mode
        """

        print("\n---- Training network ...")

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        print("\n---- Timestamp: {}".format(self.timestamp))

        # load to gpu (if possible)
        # pylint: disable=self-cls-assignment
        self, data, _ = utils.move_to_gpu(self, data)

        # data loader
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.params)
        optimizer = opt.SGD(
            self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"]
        )
        if hasattr(self, "optimizer_state_dict"):
            optimizer.load_state_dict(self.optimizer_state_dict)

        # training scheduler
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer)

        best_loss = -1
        self.losses = {"train_loss": [], "val_loss": [], "test_loss": []}
        for epoch in range(
            self.params.get("epoch", 0), self.params.get("epoch", 0) + self.params["epochs"]
        ):
            self._epoch = epoch

            train_loss, optimizer = self.batch_loss(
                data, train_loader, train=True, verbose=verbose, optimizer=optimizer
            )
            val_loss, _ = self.batch_loss(data, val_loader, verbose=verbose)
            scheduler.step(train_loss)

            print(
                f"\nEpoch: {self._epoch}, Training loss: {train_loss:4f}, Validation loss: {val_loss:.4f}, lr: {scheduler._last_lr[0]:.4f}",  # noqa, pylint: disable=line-too-long,protected-access
                end="",
            )

            if best_loss == -1 or (val_loss < best_loss):
                outdir = self.save_model(
                    optimizer, self.losses, outdir=outdir, best=True, timestamp=self.timestamp
                )
                best_loss = val_loss
                print(" *", end="")

            self.losses["train_loss"].append(train_loss)
            self.losses["val_loss"].append(val_loss)

        test_loss, _ = self.batch_loss(data, test_loader)
        print(f"\nFinal test loss: {test_loss:.4f}")

        self.losses["test_loss"].append(test_loss)

        self.save_model(optimizer, self.losses, outdir=outdir, best=False, timestamp=self.timestamp)
        self.load_model(os.path.join(outdir, f"best_model_{self.timestamp}.pth"))

    def load_model(self, loadpath):
        """Load model.

        Args:
            loadpath: directory with models to load best model, or specific model path
        """
        checkpoint = torch.load(
            loadpath, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._epoch = checkpoint["epoch"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        if hasattr(self, "losses"):
            self.losses = checkpoint["losses"]

    def save_model(self, optimizer, losses, outdir=None, best=False, timestamp=""):
        """Save model."""
        if outdir is None:
            outdir = "./outputs/"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        checkpoint = {
            "epoch": self._epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "time": timestamp,
            "params": self.params,
            "losses": losses,
        }

        if best:
            fname = "best_model_"
        else:
            fname = "last_model_"

        fname += timestamp
        fname += ".pth"

        if best:
            torch.save(checkpoint, os.path.join(outdir, fname))
        else:
            torch.save(checkpoint, os.path.join(outdir, fname))

        return outdir


class loss_fun(nn.Module):
    """Loss function."""    

    def forward(self, out, mask=None, data=None, n_id=None, n_batch=None):
        """forward."""
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()  # pylint: disable=not-callable
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()  # pylint: disable=not-callable

        coagulation_loss = 0.0
        if mask is not None:
            z_mask = out[mask]
            coagulation_loss = (z_mask - z_mask.mean(dim=0)).norm(dim=1).sum()
            
        # perform procrustes on embeddings between systems
        if data:
            embeddings, indices = group_embeddings_by_graph(n_id[:n_batch], data.batch, out, limit_rows=False)
            emb_dist = distance(embeddings, dist_type='procrustes')
            #emb_dist = distance(embeddings, dist_type='mmd')

        #print(emb_dist)
        return -pos_loss - neg_loss + torch.sigmoid(coagulation_loss) - 0.5 + 0.1*emb_dist 
            

class ortho_loss(nn.Module):
    """ custom loss based on orthogonal transform distance """
    
    def forward(self, out, data=None, n_id=None, n_batch=None):
        embeddings, indices = group_embeddings_by_graph(n_id[:n_batch], data.batch, out, limit_rows=False)
        emb_dist = distance(embeddings, dist_type='procrustes', return_paired=True)
        return emb_dist 
    

def group_embeddings_by_graph(target_id, graph_ids, emb, limit_rows=True):
    embs = [emb[graph_ids[target_id] == gid]  for gid in graph_ids.unique().tolist()]
    indices = [(graph_ids[target_id] == gid).nonzero() for gid in graph_ids.unique().tolist()]

    # procrustes requires us to have same size matrices
    if limit_rows:
        max_rows = min([u.shape[0] for u in embs])
        return [emb[:max_rows,:] for emb in embs], indices
    else:
        return embs, indices

def euclidean_distance(a, b):
    return torch.norm(a - b, dim=1)  # Compute Euclidean distance

def distance(embeddings, dist_type='procrustes', return_paired=False):    
    distances = torch.zeros([len(embeddings),len(embeddings)])
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):  # Avoid redundant calculations
            if dist_type=='mmd':
                dist = mmd_distance(embeddings[i], embeddings[j])
            if dist_type=='procrustes':
                dist = orthogonal_procrustes_distance(embeddings[i], embeddings[j])
                #dist = affine_transform_distance(embeddings[i], embeddings[j])
            distances[i,j] = dist
    distances = distances + distances.T
    if return_paired:
        return distances.mean(axis=0)
    else:
        return distances.max()



def gaussian_kernel(x, y, sigma=1.0):
    beta = 1. / (2. * sigma ** 2)
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-beta * dist)

def mmd_distance(x, y, sigma=1.0):
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()


def affine_transform_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # match sizes for affine transform
    size = min(x.shape[0], y.shape[0])
    x = x[:size, :]
    y = y[:size, :]

    # Ensure x is augmented with ones to account for translation
    ones = torch.ones(x.shape[0], 1, device=x.device)
    x_augmented = torch.cat([x, ones], dim=1)

    # Solve for the best affine transformation A_augmented
    # where A_augmented * x_augmented ≈ y
    result = torch.linalg.lstsq(x_augmented, y)
    A_augmented = result.solution[:x_augmented.shape[1], :]  # Truncate extra rows if necessary

    # Separate the linear transformation (A) and translation (b) components
    A = A_augmented[:-1, :]
    b = A_augmented[-1, :]

    # Enforce constraints to prevent scaling
    # For rotation and reflection: ensure orthogonal columns with unit norm
    U, _, V = torch.linalg.svd(A, full_matrices=False)
    A_no_scale = torch.matmul(U, V.t())

    # For shearing: Modify A_no_scale to include shear factors, ensuring no scaling is introduced
    # This can be complex as it depends on how you define shearing, and may require additional parameters or assumptions

    # Apply the constrained affine transformation to x
    x_transformed = torch.matmul(x, A_no_scale) + b

    # Compute the distance as the Frobenius norm of the difference between the transformed x and y
    distance = torch.linalg.norm(x_transformed - y, ord='fro')
    #distance = mmd_distance(x_transformed, y)

    return distance



def orthogonal_procrustes_distance(x: torch.Tensor,
                                   y: torch.Tensor,
                                   ) -> torch.Tensor:
    """
    Computes Procrustes distance between representations A and B using PyTorch tensors.
    """
    
    x = _matrix_normalize(x, dim=0)
    y = _matrix_normalize(y, dim=0)
    
    size = min([x.shape[0],y.shape[0]])
    x = x[:size,:]
    y = y[:size,:]
    
    # Compute squared Frobenius norms of A and B
    x_sq_frob = torch.sum(x ** 2)
    y_sq_frob = torch.sum(y ** 2)

    # Compute nuclear norm of the product of A and B.T
    nuc = torch.linalg.norm(x @ y.T, ord='nuc')  # Note: 'nuc' stands for nuclear norm

    # Calculate and return the Procrustes distance
    return x_sq_frob + y_sq_frob - 2 * nuc



def orthogonal_procrustes_distance_(x: torch.Tensor,
                                   y: torch.Tensor,
                                   ) -> torch.Tensor:
    """ Orthogonal Procrustes distance used in Ding+21.
    Returns in dist interval [0, 1].

    Note:
        -  for a raw representation A we first subtract the mean value from each column, then divide
    by the Frobenius norm, to produce the normalized representation A* , used in all our dissimilarity computation.
        - see uutils.torch_uu.orthogonal_procrustes_distance to see my implementation
    Args:
        x: input tensor of Shape NxD1
        y: input tensor of Shape NxD2
    Returns:
    """
    # _check_shape_equal(x, y, 0)
    nuclear_norm = partial(torch.linalg.norm, ord="nuc")

    x = _matrix_normalize(x, dim=0)
    y = _matrix_normalize(y, dim=0)
    
    size = min([x.shape[0],y.shape[0]])
    x = x[:size,:]
    y = y[:size,:]
    # note: ||x||_F = 1, ||y||_F = 1
    # - note this already outputs it between [0, 1] e.g. it's not 2 - 2 nuclear_norm(<x1, x2>) due to 0.5*d_proc(x, y)
    return 1 - nuclear_norm(x.t() @ y)

def _zero_mean(input: torch.Tensor,
               dim: int
               ) -> torch.Tensor:
    return input - input.mean(dim=dim, keepdim=True)

def _matrix_normalize(input: torch.Tensor,
                      dim: int
                      ) -> torch.Tensor:
    """
    Center and normalize according to the forbenius norm of the centered data.

    Note:
        - this does not create standardized random variables in a random vectors.
    ref:
        - https://stats.stackexchange.com/questions/544812/how-should-one-normalize-activations-of-batches-before-passing-them-through-a-si
    :param input:
    :param dim:
    :return:
    """
    from torch.linalg import norm
    X_centered: torch.Tensor = _zero_mean(input, dim=dim)
    X_star: torch.Tensor = X_centered / norm(X_centered, "fro")
    return X_star


def generalized_procrustes_tensor(
    tensor_list: List[torch.Tensor],
    tol: float = 1e-7,
    n_iter: int = 200
) -> Tuple[List[torch.Tensor], float]:
    """Generalized Procrustes Analysis for PyTorch tensors.

    Parameters
    ----------
    tensor_list : List[torch.Tensor]
        The list of 2D-tensors to be transformed.
    tol : float, optional
        Tolerance value to stop the iterations.
    n_iter : int, optional
        Number of total iterations.

    Returns
    -------
    tensor_aligned : List[torch.Tensor]
        A list of transformed tensors with generalized Procrustes analysis.
    new_distance_gpa : float
        The distance for matching all the transformed tensors with generalized Procrustes analysis.
    """
    if n_iter <= 0:
        raise ValueError("Number of iterations should be a positive number.")

    # Initialize with the first tensor as the reference
    ref = tensor_list[0].clone()
    tensor_aligned = [ref] + [None] * (len(tensor_list) - 1)

    distance_gpa = float('inf')
    for _ in range(n_iter):
        # Align all tensors to the current reference
        for i, tensor in enumerate(tensor_list):  
            tensor_aligned[i] = _orthogonal(tensor, ref)

        # Update the reference as the mean of aligned tensors
        new_ref = torch.mean(torch.stack(tensor_aligned), dim=0)

        # Calculate the distance change for convergence check
        new_distance_gpa = torch.norm(ref - new_ref).item() ** 2
        if distance_gpa != float('inf') and abs(new_distance_gpa - distance_gpa) < tol:
            break

        ref = new_ref
        distance_gpa = new_distance_gpa
        

    return tensor_aligned, new_distance_gpa

def _orthogonal(arr_a: torch.Tensor, arr_b: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes transformation and returns the transformed array."""
    
    # Compute the matrix product of arr_b^T and arr_a
    mat = torch.matmul(arr_b.t(), arr_a)

    # Compute the singular value decomposition
    U, _, V = torch.linalg.svd(mat, full_matrices=False)

    # Compute the optimal rotation matrix R
    R = torch.matmul(U, V.t())

    # Apply the transformation
    transformed = torch.matmul(arr_a, R)

    return transformed

