"""Main network"""
import glob
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as opt
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch_geometric.nn import MLP
from tqdm import tqdm

from MARBLE import dataloader
from MARBLE import geometry
from MARBLE import layers
from MARBLE import utils


class net(nn.Module):
    """MARBLE neural network.

    The possible parameters and their default values are described below, 
    and can be accessed via the `params` dictionnary in this class constructor.

    Parameters
    ----------
    # training parameters
    batch_size: batch size (default=64)
    epochs: optimisation epochs (default=20)
    lr: iniital learning rate (default=0.01)
    momentum: momentum (default=0.9)
    diffusion: set to True to use diffusion layer before gradient computation (default=False)
    include_positions: include positions as features (warning: this is untested!) (default=False)
    include_self: include vector at the center of feature (default=True)

    # manifold/signal parameters
    order: order to which to compute the directional derivatives (default=2)
    inner_product_features: transform gradient features to inner product features (default=True)
    frac_sampled_nb: fraction of neighbours to sample for gradient computation
        (if -1 then all neighbours) (default=-1)

    #network parameters
    dropout: dropout in the MLP (default=0.)
    n_lin_layers: number of layers if MLP (default=2)
    hidden_channels: number of hidden channels (default=16)
    out_channels: number of output channels (if null, then =hidden_channels) (default=3)
    bias: learn bias parameters in MLP (default=True)
    vec_norm: normalise features to unit length (default=False)
    batch_norm: batch normalisation (default=False)

    #other params
    seed: seed for reproducibility (default=0)
    processes: number of cpus (default=1)
    """

    def __init__(self, data, loadpath=None, params=None, verbose=True):
        """
        Constructor of the MARBLE net.

        Parameters
        ----------
        data: PyG data
        loadpath: path to a model file, or a directory with models (best model will be used)
        params: can be a dict with parameters to overwrite default params or a path to a yaml file
        verbose: run in verbose mode
        """
        super().__init__()

        if loadpath is not None:
            if Path(loadpath).is_dir():
                loadpath = max(glob.glob(f"{loadpath}/best_model*"))
            self.params = torch.load(loadpath)["params"]
        else:
            if params is not None:
                if isinstance(params, str) and Path(params).exists():
                    with open(params, "rb") as f:
                        params = yaml.safe_load(f)
                self.params = params

        self._epoch = 0  # to resume optimisation
        self.parse_parameters(data)
        self.check_parameters(data)
        self.setup_layers()
        self.loss = loss_fun()
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

        if data.local_gauges:
            assert self.params[
                "inner_product_features"
            ], "Local gauges detected, so >>inner_product_features<< must be True"

        pars = [
            "batch_size",
            "epochs",
            "lr",
            "momentum",
            "order",
            "inner_product_features",
            "dim_signal",
            "dim_emb",
            "dim_man",
            "frac_sampled_nb",
            "dropout",
            "n_lin_layers",
            "diffusion",
            "hidden_channels",
            "out_channels",
            "bias",
            "batch_norm",
            "vec_norm",
            "seed",
            "n_sampled_nb",
            "processes",
            "include_positions",
            "include_self",
        ]

        for p in self.params.keys():
            assert p in pars, f"Unknown specified parameter {p}!"

    def reset_parameters(self):
        """reset parmaeters."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def setup_layers(self):
        """Setup layers."""

        s, d, o = self.params["dim_signal"], self.params["dim_emb"], self.params["order"]
        if "dim_man" in self.params.keys():
            s = d = self.params["dim_man"]

        # diffusion
        self.diffusion = layers.Diffusion()

        # gradient features
        self.grad = nn.ModuleList(layers.AnisoConv(self.params["vec_norm"]) for i in range(o))

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
            cum_channels += d

        if not self.params["include_self"]:
            cum_channels -= s

        # encoder
        channel_list = (
            [cum_channels]
            + (self.params["n_lin_layers"] - 1) * [self.params["hidden_channels"]]
            + [self.params["out_channels"]]
        )

        self.enc = MLP(
            channel_list=channel_list,
            dropout=self.params["dropout"],
            norm=self.params["batch_norm"],
            bias=self.params["bias"],
        )

    def forward(self, data, n_id, adjs=None):
        """Forward pass.
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to
        simplify message passing. By convention, the first size[1] entries of x
        are the target nodes, i.e, x = concat[x_target, x_other]."""

        x = data.x
        n, d = data.x.shape[0], data.gauges.shape[2]

        # local gauges
        if self.params["inner_product_features"]:
            x = geometry.map_to_local_gauges(x, data.gauges)

        # diffusion
        if self.params["diffusion"]:
            L = data.L.copy() if hasattr(data, "L") else None
            Lc = data.Lc.copy() if hasattr(data, "Lc") else None
            x = self.diffusion(x, L, Lc=Lc, method="spectral")

        # restrict to current batch
        x = x[n_id]
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
            out.append(x)

        # take target nodes
        out = [o[: size[1]] for o in out]  # pylint: disable=undefined-loop-variable

        # inner products
        if self.params["inner_product_features"]:
            out = self.inner_products(out)
        else:
            out = torch.cat(out, axis=1)

        if self.params["include_positions"]:
            out = torch.hstack(
                [data.pos[n_id[: size[1]]], out]  # pylint: disable=undefined-loop-variable
            )

        emb = self.enc(out)

        return emb

    def evaluate(self, data):
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

            # load to gpu if possible
            (
                _,
                data.x,
                data.pos,
                data.L,
                data.Lc,
                data.kernels,
                data.gauges,
                adjs,
            ) = utils.move_to_gpu(self, data, adjs)

            out = self.forward(data, torch.arange(len(data.x)), adjs)

            (
                _,
                data.x,
                data.pos,
                data.L,
                data.Lc,
                data.kernels,
                data.gauges,
                adjs,
            ) = utils.detach_from_gpu(self, data, adjs)

            data.emb = out.detach().cpu()

            return data

    def batch_loss(self, data, loader, train=False, verbose=False, optimizer=None):
        """Loop over minibatches provided by loader function.

        Parameters
        ----------
        x : (nxdim) feature matrix
        loader : dataloader object from dataloader.py

        """

        if train:  # training mode (enables dropout in MLP)
            self.train()

        if verbose:
            print("\n")

        cum_loss = 0
        for batch in tqdm(loader, disable=not verbose):
            _, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]

            emb = self.forward(data, n_id, adjs)
            loss = self.loss(emb)
            cum_loss += float(loss)

            if optimizer is not None:
                optimizer.zero_grad()  # zero gradients, otherwise accumulates
                loss.backward()  # backprop
                optimizer.step()

        self.eval()

        return cum_loss / len(loader), optimizer

    def run_training(self, data, outdir=None, verbose=False):
        """Network training.

        Parameters
        ----------
        data: PyG data
        outdir: folder to save intermediate models
        verbose: run in verbose mode
        """

        print("\n---- Training network ...")

        time = datetime.now().strftime("%Y%m%d-%H%M%S")

        # load to gpu (if possible)
        # pylint: disable=self-cls-assignment
        self, data.x, data.pos, data.L, data.Lc, data.kernels, data.gauges, _ = utils.move_to_gpu(
            self, data
        )

        # data loader
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.params)
        optimizer = opt.SGD(
            self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"]
        )
        if hasattr(self, "optimizer_state_dict"):
            optimizer.load_state_dict(self.optimizer_state_dict)
        writer = SummaryWriter("./log/")

        # training scheduler
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer)

        best_loss = -1
        for epoch in range(
            self.params.get("epoch", 0), self.params.get("epoch", 0) + self.params["epochs"]
        ):
            self._epoch = epoch

            train_loss, optimizer = self.batch_loss(
                data, train_loader, train=True, verbose=verbose, optimizer=optimizer
            )
            val_loss, _ = self.batch_loss(data, val_loader, verbose=verbose)
            scheduler.step(train_loss)

            writer.add_scalar("Loss/train", train_loss, self._epoch)
            writer.add_scalar("Loss/validation", val_loss, self._epoch)
            writer.flush()
            print(
                f"\nEpoch: {self._epoch}, Training loss: {train_loss:4f}, Validation loss: {val_loss:.4f}, lr: {scheduler._last_lr[0]:.4f}",  # noqa, pylint: disable=line-too-long,protected-access
                end="",
            )

            if best_loss == -1 or (val_loss < best_loss):
                outdir = self.save_model(optimizer, outdir, best=True, timestamp=time)
                best_loss = val_loss
                print(" *", end="")

        test_loss, _ = self.batch_loss(data, test_loader)
        writer.add_scalar("Loss/test", test_loss)
        writer.close()
        print(f"\nFinal test loss: {test_loss:.4f}")

        self.save_model(optimizer, outdir, best=False, timestamp=time)

        self.load_model(os.path.join(outdir, f"best_model_{time}.pth"))

    def load_model(self, loadpath):
        """Load model.

        Parameters
        ----------
        loadpath: directory with models to load best model, or specific model path
        """
        checkpoint = torch.load(loadpath)
        self._epoch = checkpoint["epoch"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]

    def save_model(self, optimizer, outdir=None, best=False, timestamp=""):
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

    def forward(self, out):
        """forward."""
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()

        return -pos_loss - neg_loss
