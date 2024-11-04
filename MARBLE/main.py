"""Main network"""

import glob
import os
import warnings
from datetime import datetime
from pathlib import Path

import ot
import torch
import torch.nn.functional as F
import torch.optim as opt
import yaml
from torch import nn
from torch_geometric.nn import MLP, Linear
from torch.nn.utils.parametrizations import orthogonal
from tqdm import tqdm

from MARBLE import dataloader
from MARBLE import geometry
from MARBLE import layers
from MARBLE import utils


class net(nn.Module):
    """MARBLE neural network.

    The possible parameters and their default values are described below,
    and can be accessed via the `params` dictionnary in this class constructor.

    Args:
        batch_size (int): Batch size for training. Default is 64.
        epochs (int): Number of optimization epochs. Default is 20.
        lr (float): Initial learning rate. Default is 0.01.
        momentum (float): Momentum for the optimizer. Default is 0.9.
        diffusion (bool): If True, use a diffusion layer before gradient computation. Default is False.
        include_positions (bool): Include positions as features. Default is False.
        include_self (bool): Include vector at the center of the feature. Default is True.
        order (int): Order to compute the directional derivatives. Default is 2.
        inner_product_features (bool): Transform gradient features to inner product features. Default is True.
        frac_sampled_nb (float): Fraction of neighbors to sample for gradient computation.
            If -1, all neighbors are used. Default is -1.
        dropout (float): Dropout rate in the MLP. Default is 0.0.
        hidden_channels (int or list): Number of hidden channels. If a list is provided, multiple layers are added. Default is 16.
        out_channels (int): Number of output channels. If None, defaults to hidden_channels. Default is 3.
        bias (bool): Whether to learn bias parameters in the MLP. Default is True.
        vec_norm (bool): Normalize features at each derivative order to unit length. Default is False.
        emb_norm (bool): Normalize the MLP output to unit length. Default is False.
        batch_norm (bool): Apply batch normalization. Default is True.
        seed (int): Seed for reproducibility.
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
        self.alignment_loss = AlignmentLossFun()
        self.representation_loss = RepresentationLossFun()
        self.reset_parameters()
        self.timestamp = None
        
        self.w = nn.Parameter(torch.tensor(0.0))

        if verbose:
            utils.print_settings(self)

        if loadpath is not None:
            self.load_model(loadpath)
            

    def parse_parameters(self, data):
        """Load default parameters from a YAML file and merge them with user-specified parameters."""

        # Load default parameters from YAML file
        default_file = os.path.join(os.path.dirname(__file__), "default_params.yaml")
        with open(default_file, "rb") as f:
            default_params = yaml.safe_load(f)
        
        # Update parameters with data-specific values
        default_params["dim_signal"] = data.x.shape[1]
        default_params["dim_emb"] = data.pos.shape[1]
        default_params["slices"] = data._slice_dict["x"]

        if hasattr(data, "dim_man"):
            default_params["dim_man"] = data.dim_man

        # Merge default parameters with user-specified parameters
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

        # Calculate the number of sampled neighbors if a fraction is specified
        if default_params["frac_sampled_nb"] != -1:
            self.params["n_sampled_nb"] = int(data.degree * default_params["frac_sampled_nb"])
        else:
            self.params["n_sampled_nb"] = -1

        self.params["batch_norm"] = "batch_norm" if self.params["batch_norm"] else None


    def check_parameters(self, data):
        """Check parameter validity"""

        assert self.params["order"] > 0, "Derivative order must be at least 1!"

        if self.params["vec_norm"]:
            assert data.x.shape[1] > 1, "Using vec_norm=True is not permitted for scalar signals"

        if self.params["diffusion"]:
            assert hasattr(data, "L"), "No Laplacian found. Compute it in preprocessing()!"

        pars = [
            "align_datasets",
            "architecture",
            "batch_norm",
            "batch_size",
            "bias",
            "dim_emb",
            "dim_signal",
            "diffusion",
            "dropout",
            "emb_norm",
            "epochs",
            "frac_sampled_nb",
            "GAT_hidden_layers",
            "GAT_attention_heads",
            "hidden_channels",
            "include_positions",
            "include_self",
            "inner_product_features",
            "lr",
            "momentum",
            "order",
            "out_channels",
            "seed",
            "vec_norm",
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
            s = d = self.params["dim_man"]

        # diffusion
        self.diffusion = layers.Diffusion()

        # gradient features
        if self.params['architecture']=='gradient_filter':
            self.grad = nn.ModuleList(layers.AnisoConv() for i in range(o))
        elif self.params['architecture']=='GAT':
            self.GCN = nn.ModuleList([layers.GCN(), layers.GCN() ])
        else:
            NotImplementedError()

        # cumulated number of channels after gradient features
        if self.params['architecture']=='gradient_filter':
            cum_channels = s * (1 - d ** (o + 1)) // (1 - d)
        elif self.params['architecture']=='GAT':
            cum_channels = s*(o+1)
        else:
            NotImplementedError()
            
        if not self.params["include_self"]:
            cum_channels -= s

        if self.params["inner_product_features"]:
            cum_channels //= s
            if s == 1:
                cum_channels = o + 1
            self.inner_products = layers.InnerProductFeatures(cum_channels, s)
        else:
            self.inner_products = None

        if self.params["include_positions"]:
            cum_channels += d

        # encoder
        if not isinstance(self.params["hidden_channels"], list):
            self.params["hidden_channels"] = [self.params["hidden_channels"]]
        
        # self.maps = nn.ModuleList((Linear(s, s,
        #                             bias=self.params["bias"],
        #                             ))
        #                             for i in range(len(self.params['slices'])-2))
                                    # for i in range(len(self.params['slices'])-3))
        self.maps = nn.ModuleList(layers.ParametrizedRotation()
                                    for i in range(len(self.params['slices'])-2))
        
        # for i, m in enumerate(self.maps):
        #     torch.nn.init.constant_(m.bias, 0)
        #     torch.nn.init.eye_(m.weight)
        #     self.maps[i] = orthogonal(m)
                                            
        # self.maps = nn.ModuleList(MLP(
        #                             channel_list=[s, 2*s, s],
        #                             dropout=self.params["dropout"],
        #                             bias=self.params["bias"],
        #                             norm=self.params["batch_norm"],
        #                             )
        #                             for i in range(len(self.params['slices'])-1)
        # )
        
        channel_list = (
            [cum_channels] + self.params["hidden_channels"] + [self.params["out_channels"]]
        )

        self.enc = MLP(
            channel_list=channel_list,
            # dropout=self.params["dropout"],
            bias=self.params["bias"],
            norm=None#self.params["batch_norm"],
        )


    def forward(self, data, n_id, adjs=None):
        """Forward pass.
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to
        simplify message passing. By convention, the first size[1] entries of x
        are the target nodes, i.e, x = concat[x_target, x_other]."""

        x = data.x
        pos = data.pos
        n, d = x.shape[0], data.gauges.shape[2]
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
        if self.params["inner_product_features"]:
            x = geometry.global_to_local_frame(x, data.gauges)
        
        #alignment across data slices
        if self.params['align_datasets']:
            x_transform, pos_transform = [], []
            slices = self.params['slices']
            for i in range(len(slices)-1):
                pos_t_slice = pos[slices[i]:slices[i+1]]
                x_t_slice = x[slices[i]:slices[i+1]]
                if i > 0:
                # if i == 1:
                    # if self.maps[i-1].weight.min()<0:
                    #     self.maps[i-1].weight = -1*self.maps[i-1].weight
                    # if torch.det(self.maps[i-1].weight)>0:
                    #     self.maps[i-1].weight[...,0] = -1*self.maps[i-1].weight[...,0]
                    pos_t_slice, x_t_slice = geometry.transform_vector_field(
                        self.maps[i-1], pos_t_slice, x_t_slice
                    )
                    
                pos_transform.append(pos_t_slice)
                x_transform.append(x_t_slice)
                
            x_transform = torch.vstack(x_transform)
            pos_transform = torch.vstack(pos_transform)
        else:
            x_transform, pos_transform = x, pos
            
        pos, x = pos_transform, x_transform
                   
        # restrict to current batch
        x = x[n_id]
        mask = mask[n_id]
        
        # if kernels are nd x nd, meaning they act in the tangent bundle
        if data.kernels[0].size(0) == n * d:
            n_id = utils.expand_index(n_id, d)
        # is kernels are n x n meaning they act on manifold points
        else:
            d = 1

        if self.params["vec_norm"]:
            x = F.normalize(x, dim=-1, p=2)

        # gradients
        if self.params["include_self"]:
            out = [x]
        else:
            out = []
            
        for i, (edge_index, e_id, size) in enumerate(adjs):
            # graph convolutional architecture
            if self.params['architecture']=='gradient_filter':
                kernels = [K[n_id[: size[1] * d], :][:, n_id[: size[0] * d]] for K in data.kernels]
                x = self.grad[i](x, kernels)
            elif self.params['architecture']=='GAT':
                x = self.GCN[i](x, edge_index)
            else:
                NotImplementedError()

            if self.params["vec_norm"]:
                x = F.normalize(x, dim=-1, p=2)

            out.append(x)
            
        # take target nodes
        last_size = adjs[-1][2]
        out = [o[: last_size[1]] for o in out]

        # inner products
        if self.params["inner_product_features"]:
            out = self.inner_products(out)
        else:
            out = torch.cat(out, axis=1)

        if self.params["include_positions"]:
            out = torch.hstack([data.pos[n_id[: last_size[1]]], out])
            
        # positional_encoding = 1
        # if positional_encoding:
        #     out = torch.hstack([out, data.L[1][n_id[: last_size[1]], :5]])

        # map to latent space
        emb = self.enc(out)

        # spherical output
        if self.params["emb_norm"]: 
            emb = F.normalize(emb)

        return emb, mask[: last_size[1]], n_id[:last_size[1]], pos_transform, x_transform


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
            out, _, _, data.pos_transform, data.x_transform = self.forward(data, torch.arange(len(data.x)), adjs)
            utils.detach_from_gpu(self, data, adjs)

            data.emb = out.detach().cpu()

            return data


    def batch_loss(self, data, loader, loss_fun, train=False, verbose=False, optimizer=None):
        """Loop over minibatches provided by the loader function.

        Args:
            x : (nxdim) feature matrix
            loader : dataloader object from dataloader.py

        """

        if train:  # training mode (enables dropout in MLP)
            self.train()

        if verbose:
            print("\n")

        cum_loss = 0.0
        
        # Loop over batches in the loader
        for batch in tqdm(loader, disable=not verbose):
            _, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]
            
            # Forward pass
            emb, mask, n_target, _, _ = self.forward(data, n_id, adjs)
            
            # Calculate loss
            loss = loss_fun(emb, self._epoch, list(self.named_parameters())[0][1], mask, n_target, self.params['slices'])
            cum_loss += float(loss)

            # Backpropagation and optimization if in training mode
            if optimizer is not None:
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()        # Backpropagate the loss
                optimizer.step()       # Update model parameters
                
            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad}")
                
        self.eval() # Set model to evaluation mode

        return cum_loss / len(loader), optimizer


    def run_training(self, data, outdir=None, verbose=False):
        """Run training."""
        warnings.warn("MARBLE.run_training() is deprecated. Use MARBLE.fit() instead.")

        self.fit(data, outdir=outdir, verbose=verbose)
        
        
    def optimisation_loop(self, data, train_loader, val_loader, loss_fn, optimizer, scheduler=None, outdir=None, verbose=False):
        best_loss = float('inf')
        losses = {"train_loss": [], "val_loss": [], "test_loss": []}
        
        # Epoch loop
        for epoch in range(self.params.get("epoch", 0), self.params.get("epoch", 0) + self.params["epochs"]):
            self._epoch = epoch

            # Training loss calculation
            train_loss, optimizer = self.batch_loss(
                data, train_loader, loss_fn, train=True, verbose=verbose, optimizer=optimizer
            )
            
            # Validation loss calculation
            val_loss, _ = self.batch_loss(data, val_loader, loss_fn, verbose=verbose)
            
            # Step the scheduler based on the training loss
            if scheduler is not None:
                scheduler.step(train_loss)

            # Log current losses and learning rate
            log_message = f"\nEpoch: {self._epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}"
            if scheduler is not None:
                log_message += f", lr: {scheduler._last_lr[0]:.4f}"
            print(log_message, end="")

            # Save the model if validation loss improves
            if val_loss < best_loss:
                outdir = self.save_model(optimizer, losses, outdir=outdir, best=True, timestamp=self.timestamp)
                best_loss = val_loss
                print(" *", end="")

            losses["train_loss"].append(train_loss)
            losses["val_loss"].append(val_loss)
            
        return optimizer, losses, outdir
    

    def fit(self, data, outdir=None, verbose=False):
        """Network training.

        Args:
            data: PyG data
            outdir: folder to save intermediate models
            verbose: run in verbose mode
        """

        print("\n---- Training network ...")

        # Set timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n---- Timestamp: {self.timestamp}")

        # Move model and data to GPU (if available)
        self, data, _ = utils.move_to_gpu(self, data)

        # data loader
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.params)
        
        # Optimizers and scheduler
        # optimizer_rep = opt.SGD(self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])
        # scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer_rep)
        # optimizer_ali = opt.SGD(self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])
        
        # Load optimizer state if available
        # if hasattr(self, "optimizer_state_dict"):
        #     optimizer_rep.load_state_dict(self.optimizer_state_dict)        
        
        # Training loop
        for _ in range(1):
            
            # Alignment learning
            print('\n\nRepresentation alignment')
            
            for n, p in self.named_parameters():
                if 'maps' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
                # print(n,p)
                    
            optimizer_ali = opt.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.1, momentum=0.9)

            optimizer_ali, losses, outdir = self.optimisation_loop(
                data, train_loader, val_loader, self.alignment_loss, optimizer_ali, outdir=outdir, verbose=verbose
            )
            
            
            # Representation learning
            print('\n\nRepresentation learning')
            
            for n, p in self.named_parameters():
                if 'maps' in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    
            optimizer_rep = opt.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.params["lr"], momentum=self.params["momentum"])
            scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer_rep)
                    
            optimizer_rep, losses, outdir = self.optimisation_loop(
                data, train_loader, val_loader, self.representation_loss, optimizer_rep, 
                scheduler=scheduler, outdir=outdir, verbose=verbose
            )
            
            # from MARBLE import postprocessing, plotting
            # data = self.transform(data)
            # data.emb_2D, manifold = geometry.embed(data.emb, seed=0)
            # plotting.embedding(data, data.y.numpy(), titles=None)
            
            
           
            
            # data = self.transform(data)
            # data.emb_2D, _ = geometry.embed(data.emb, manifold=manifold, seed=0)
            # plotting.embedding(data, data.y.numpy(), titles=None)
            
            
            
        # Test representation loss computation
        test_loss, _ = self.batch_loss(data, test_loader, self.representation_loss)
        print(f"\nFinal test loss: {test_loss:.4f}")
        
        # Save test loss and model
        losses["test_loss"].append(test_loss)
        self.save_model(optimizer_ali, losses, outdir=outdir, best=False, timestamp=self.timestamp)
        
        # Load the best model
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
        self.load_state_dict(checkpoint["model_state_dict"], strict=False)
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


class RepresentationLossFun(nn.Module):
    """Loss function measuring the representation quality."""
    
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, out, epoch, w=None, mask=None, n_target=None, slices=None):
        
        # Loss measuring the similarity of embedded latent features that were adjacent in data space
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()  # pylint: disable=not-callable
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()  # pylint: disable=not-callable
        
        unsupervised_loss = -pos_loss - neg_loss

        # Loss ensuring that specific points defined by mask are embedded close in latent space
        coagulation_loss = 0.0
        if mask is not None:
            z_mask = out[mask]
            coagulation_loss = (z_mask - z_mask.mean(dim=0)).norm(dim=1).sum()
            
        coagulation_loss = torch.sigmoid(coagulation_loss) - 0.5
            
        return unsupervised_loss + coagulation_loss
    
    
class AlignmentLossFun(nn.Module):
    """Loss function measuring the alignment between datasets."""
    
    def __init__(self):
        super().__init__()

    def forward(self, out, epoch, w, mask=None, n_target=None, slices=None):       
        distr, slices_batch, bins = [], [0], []
        n_datasets = len(slices) - 1
                
        for i in range(n_datasets):
            idx = (n_target >= slices[i]) & (n_target < slices[i + 1])
            n = int(idx.sum())
            if n == 0:
                # Return 0 loss if no sample is drawn from a dataset
                return torch.tensor(0.0, requires_grad=True)
            
            slices_batch.append(n + slices_batch[-1])
            distr.append(out[idx].unsqueeze(0))
            
            bin_size = slices_batch[i + 1] - slices_batch[i]
            bins.append(torch.ones(bin_size) / bin_size)
                        
        distr_all = torch.cat(distr, axis=1)
        
        cdists =  torch.squeeze(torch.cdist(distr_all, distr_all))
        
        alignment_loss = 0.0
        for i in range(n_datasets-1):
            for j in range(i+1, n_datasets):
                dists = cdists[slices_batch[i]:slices_batch[i+1], 
                               slices_batch[j]:slices_batch[j+1]]
                alignment_loss += ot.emd2(bins[i], bins[j], dists)
        
        return alignment_loss
 