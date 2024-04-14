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

from typing import List, Tuple, Optional

import matplotlib.pyplot as plt

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
        self.initial_rotations = data.initial_rotations
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
        params["n_systems"] = data.num_systems
        params["n_conditions"] = data.num_conditions

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
            
        if self.params["n_systems"] == 1:
            print("\n---- Only one system: setting global alignment to False", end="")
            self.params["global_align"] = False

    def check_parameters(self, data):
        """Check parameter validity"""

        assert self.params["order"] > 0, "Derivative order must be at least 1!"

        if self.params["vec_norm"]:
            assert data.x.shape[1] > 1, "Using vec_norm=True is not permitted for scalar signals"

        if self.params["scalar_diffusion"]:
            assert hasattr(data, "L"), "No Laplacian found. Compute it in preprocessing()!"
        
        if self.params["vector_diffusion"]:
            assert hasattr(data, "Lc"), "No Laplacian found. Compute it in preprocessing()!"

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
            "scalar_diffusion",
            "vector_diffusion",
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
        
        # add list of orthogonal transform layers - one for each system
        if self.params['global_align']:
            self.orthogonal_transform = nn.ModuleList([
                                            layers.OrthogonalTransformLayer(s, self.initial_rotations[i]) for i in range(self.params["n_systems"])
                                        ])
            self.orthogonal_transform[0].Q = nn.Parameter(torch.eye(s))
            self.orthogonal_transform[0].Q.requires_grad = False # fix the first system from rotating.
            


    def forward(self, data, n_id, adjs=None, n_batch=None):
        """Forward pass.
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to
        simplify message passing. By convention, the first size[1] entries of x
        are the target nodes, i.e, x = concat[x_target, x_other]."""

        x = data.x
        n, d = x.shape[0], data.gauges.shape[2]
        dim_man = data.gauges.shape[2]
        dim_space = data.x.shape[1]
        mask = data.mask

        # diffusion
        if self.params["vector_diffusion"]:
            #if hasattr(data, "Lc"):                
            # rotate local gauges if performing vector diffusion ... 
            if self.params['gauge_grad']:
                l_gauges_s, indices = geometry.split_by_system(torch.arange(len(data.x)), data.system, data.l_gauges) # limit_rows=False)
                rotated_gauges = [self.orthogonal_transform[i](l_gauges_s[i].view(-1, dim_space)).view(-1, l_gauges_s[i].shape[1], l_gauges_s[i].shape[2]) for i in range(len(l_gauges_s))]    
                l_gauges = torch.zeros_like(data.l_gauges)
                l_gauges[torch.cat(indices, dim=0).squeeze()] = torch.cat(rotated_gauges, dim=0) 
                l_gauges = l_gauges.detach()                    
            
            # perform vector diffusion
            x = geometry.global_to_local_frame(x, l_gauges)
            x = self.diffusion(x, data.L, Lc=data.Lc, method="spectral")
            x = geometry.global_to_local_frame(x, l_gauges, reverse=True)
        elif self.params['scalar_diffusion']:
            x = self.diffusion(x, data.L, method="spectral")

        # local gauges
        if self.params["inner_product_features"] or (dim_man < dim_space): # self.params['global_align'] or 
            x = geometry.global_to_local_frame(x, data.gauges)

        # restrict to current batch
        x = x[n_id]
        p = data.pos[n_id]
        mask = mask[n_id]
        n_id_orig = n_id
        if data.kernels[0].size(0) == n * d:
            n_id = utils.expand_index(n_id, d)
        else:
            d = 1

        if self.params["vec_norm"]:
            x = F.normalize(x, dim=-1, p=2)
            
        # learn orthogonal transformation before gradients?
        if self.params["global_align"]:  
            #last_size = adjs[-1][2]
            #out_ = [o[: last_size[1]] for o in out]       
            
            # transform vectors
            x_s, indices = geometry.split_by_system(n_id, data.system, x,) 
            rotated_x = [self.orthogonal_transform[i](x_s[i].view(-1,dim_space)).view(-1,x_s[i].shape[1]) for i in range(len(x_s))]    
            x = torch.zeros_like(x)
            x[torch.cat(indices, dim=0).squeeze()] = torch.cat(rotated_x, dim=0) 

            # transform positions
            p_s, indices = geometry.split_by_system(n_id, data.system, p,) 
            rotated_p = [self.orthogonal_transform[i](p_s[i].view(-1,dim_space)).view(-1,p_s[i].shape[1]) for i in range(len(p_s))]    
            p = torch.zeros_like(p)
            p[torch.cat(indices, dim=0).squeeze()] = torch.cat(rotated_p, dim=0) 
            

                
                # transform normal vectors of local tangent spaces
                # nv, indices = geometry.split_by_system(torch.arange(len(data.x)), data.system, data.normal_vectors) # limit_rows=False)
                # rotated_nv = [self.orthogonal_transform[i](nv[i].view(-1, dim_space)).view(-1, nv[i].shape[1]) for i in range(len(nv))]    
                # nv = torch.zeros_like(data.normal_vectors)
                # nv[torch.cat(indices, dim=0).squeeze()] = torch.cat(rotated_nv, dim=0) 
                # data.normal_vectors = nv
            
        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # colors = ['b','r']
        # for sys in range(2):
        # #sys = 1
        #     rot = self.orthogonal_transform[sys].Q.cpu().detach().numpy()
        #     pos = rotated_p[sys].cpu().detach().numpy()
        #     vec = rotated_x[sys].cpu().detach().numpy()
            
        #     #pos = pos @ rot
        #     #vec = vec @ rot        
        #     u, v, w = vec[:,0], vec[:,1], vec[:,2] # vectors
        #     xx, yy, zz = pos[:,0], pos[:,1], pos[:,2]  # positions            
        #     # Plot the vector field
        #     ax.quiver(xx, yy, zz, u, v, w, color=colors[sys], normalize=True)
        #     # Setting the axes labels
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')            

        
        if self.params["include_positions"] or self.params["positional_grad"]:
            out = [p]
        else:
            out = []
        
        # gradients
        if self.params["include_self"]:
            out.append(x)
        else:
            out = []
            
        for i, (_, _, size) in enumerate(adjs):
            #kernels = [K[n_id[: size[1] * d], :][:, n_id[: size[0] * d]] for K in data.kernels]
            kernels = [K[n_id, :][:, n_id] for K in data.kernels]

            x = self.grad[i](x, kernels)

            if self.params["vec_norm"]:
                x = F.normalize(x, dim=-1, p=2)

            out.append(x)

        last_size = adjs[-1][2]
        # take target nodes
        out = [o[: last_size[1]] for o in out]
            
        # need global coordinates prior to orthogonal transformations
        # if self.params["global_align"]:     
        #     for i, o in enumerate(out):
        #         # project back into ambient coordinate space
        #         # for each output only act on each local tangent direction e.g. [dx/du, dx/dv]
        #         new_o = [geometry.global_to_local_frame(o[:,d_*dim_man:d_*dim_man+dim_man], data.gauges[n_id_orig][:last_size[1]], reverse=True)
        #                  for d_ in range(int(o.shape[1]/dim_man))]
        #         new_o = torch.cat(new_o, axis=1)
        #         out[i] = new_o                            

        # inner products
        if self.params["inner_product_features"]:
            if self.params['include_positions'] or self.params['positional_grad']:
                out_inner = self.inner_products(out[1:]) # don't include positions
                out = torch.cat([out[0], out_inner], axis=1)
            else:
                out = self.inner_products(out) 
        else:
            out = torch.cat(out, axis=1)
            
        # pos = out[:,:3].cpu().detach().numpy()
        # vec = out[:,3:6].cpu().detach().numpy()
        
        # #pos = pos @ rot
        # #vec = vec @ rot        
        # u, v, w = vec[:,0], vec[:,1], vec[:,2] # vectors
        # xx, yy, zz = pos[:,0], pos[:,1], pos[:,2]  # positions
        
        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        
        # # Plot the vector field
        # ax.quiver(xx, yy, zz, u, v, w, normalize=True)
        
        # # Setting the axes labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # learn orthogonal transformation
        # if self.params["global_align"]:                
        #     o, indices = group_dd_by_system(n_id_orig[:n_batch], data.system, out, limit_rows=False)
        #     ortho = [self.orthogonal_transform[i](o[i].view(-1,dim_space)).view(-1,o[i].shape[1]) for i in range(len(o))]    
        #     out = torch.zeros_like(out)
        #     out[torch.cat(indices, dim=0).squeeze()] = torch.cat(ortho, dim=0) 

        #if self.params["include_positions"]:
        #    out = torch.hstack([p[n_id_orig[: last_size[1]]], out])           

        # remove positions from encoder embedding
        if not self.params["include_positions"] and self.params['positional_grad']:
            emb = self.enc(out[:,data.pos.shape[1]:])
        else: 
            emb = self.enc(out)
        
        if self.params["emb_norm"]:  # spherical output
            emb = F.normalize(emb)   
       
        # remove positions from the directional derivative features
        #if self.params["include_positions"]:
        #    out = out[:,data.pos.shape[1]:]
            
        return emb, mask[: last_size[1]], out

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
            emb, _, out = self.forward(data, torch.arange(len(data.x)), adjs, len(data.x))
            utils.detach_from_gpu(self, data, adjs)

            data.emb = emb.detach().cpu()
            data.out = out.detach().cpu()

            return data
        
    def transform_grad(self, data):
        """Forward pass @ custom loss """
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
        emb, _, out = self.forward(data, torch.arange(len(data.x)), adjs, len(data.x))
        #utils.detach_from_gpu(self, data, adjs)

        data.emb = emb #.detach()
        data.out = out #.detach()
        
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
            
        #self.params['final_grad'] = True
        dim_space = data.x.shape[1]

        cum_loss = 0
        cum_custom_loss = 0
                 
        if self.params['final_grad']:
            # Temporarily disable gradient computations for the orthogonal_transform layers
            original_requires_grad = [param.requires_grad for layer in self.orthogonal_transform for param in layer.parameters()]
            for layer in self.orthogonal_transform:
                for param in layer.parameters():
                    param.requires_grad = False

        # Process each batch
        for batch in tqdm(loader, disable=not verbose):            
            
            n_batch, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]
    
            # Forward pass
            emb, mask, out = self.forward(data, n_id, adjs, n_batch)
            loss = self.loss(emb, mask)
            cum_loss += float(loss)
    
            # Backward and optimize
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # Accumulates gradients for all layers
                
            # overwrite parameters on orthogonal transformation with custom loss
            if not self.params['final_grad']:
                # Compute custom loss on orthogonal transform per batch
                if self.params['global_align']:
                                        
                   
                    # compute orthogonal loss on the vectors                
                    custom_loss = self.loss_orth(out[:,dim_space:2*dim_space], data,
                                                 n_id, n_batch,
                                                 dist_type='dynamic')
                    
                    # if we include positions then use these too
                    if self.params['positional_grad']:
                        positional_loss = self.loss_orth(out[:,:dim_space], data,
                                                         n_id, n_batch,
                                                         dist_type='positional')                        
                        custom_loss = custom_loss + positional_loss
                    
                    custom_loss = custom_loss[:,0]
                    cum_custom_loss += float(custom_loss.mean())
    
                    if optimizer is not None:
                        for i, layer in enumerate(self.orthogonal_transform):
                                #optimizer.zero_grad()  # Reset gradients to zero for all model parameters                
                                for param in layer.parameters():
                                    if param.requires_grad:
                                        param.grad = torch.autograd.grad(custom_loss[i], param, retain_graph=True)[0] 
                                        #param.grad = torch.autograd.grad(custom_loss.mean(), param, retain_graph=True)[0] 
                                        #optimizer.step()
                                        
                    if optimizer is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), 0.02)      
                        optimizer.step()
    
    
        # TODO move this into preprcoessing or utils
        if not self.params['vector_grad'] and not self.params['positional_grad'] and not self.params['gauge_grad']:
            self.params['global_align'] = False
        
        # Compute custom loss for the entire dataset after all batches are processed
        custom_loss = 0
        if self.params['global_align']:
            # compute gradient and backpropagate on full set of data
            if self.params['final_grad']:
                            
                #fixed_layer = 0 #torch.randint(0, len(original_requires_grad), (1,))
                #original_requires_grad = len(original_requires_grad)*[True]
                #original_requires_grad[fixed_layer] = False
                
                # Restore requires_grad for orthogonal_transform layers
                for layer, requires_grad in zip([param for layer in self.orthogonal_transform for param in layer.parameters()], original_requires_grad):
                    layer.requires_grad = requires_grad
                
                data_ = self.transform_grad(data)  # transforms the entire dataset
                out = data_.out.to(data.x.device)
                
                # compute orthogonal loss on the vectors                  
                if self.params['vector_grad']:
                    vector_loss = self.loss_orth(out[:,dim_space:2*dim_space], data,
                                                 torch.arange(len(data.x)), len(data.x),
                                                 dist_type='dynamic', )
                    custom_loss = custom_loss + vector_loss
                    print(vector_loss.mean(axis=1))

                # if we include positions then use these too
                if self.params['positional_grad']:
                    positional_loss = self.loss_orth(out[:,:dim_space], data,
                                                     torch.arange(len(data.x)), len(data.x),
                                                     dist_type='positional',)
                    custom_loss = custom_loss + positional_loss
                    print(positional_loss.mean(axis=1))

                if self.params['gauge_grad']:
                    
                    # rotate the normal vectors...
                    nv, indices = geometry.split_by_system(torch.arange(len(data.x)), data.system, data.normal_vectors) # limit_rows=False)
                    rotated_nv = [self.orthogonal_transform[i](nv[i].view(-1, dim_space)).view(-1, nv[i].shape[1]) for i in range(len(nv))]    
                    nv = torch.zeros_like(data.normal_vectors)
                    nv[torch.cat(indices, dim=0).squeeze()] = torch.cat(rotated_nv, dim=0) 
                    #nv = data.normal_vectors
                    
                    # compute loss on gauges
                    gauge_loss = self.loss_orth(nv, data,
                                                torch.arange(len(data.x)), len(data.x),
                                                dist_type='dynamic',)
                    print(gauge_loss.mean(axis=1))
                    custom_loss = custom_loss + gauge_loss
                    
                # if self.params['derivative_grad']:
                #     derivative_loss = self.loss_orth(out[:,:dim_space], data,
                #                                      torch.arange(len(data.x)), len(data.x),
                #                                      dist_type='dynamic',)
                #     custom_loss = custom_loss + positional_loss
                    
                    
                #custom_loss = custom_loss.max(axis=1)[0] #[:,fixed_layer] # only taking the first row 
                custom_loss = custom_loss.mean(axis=1)
                cum_custom_loss += float(custom_loss.mean())
                # print(custom_loss)
                if optimizer is not None:
                    for i, layer in enumerate(self.orthogonal_transform):
                            optimizer.zero_grad()  # Reset gradients to zero for all model parameters                
                            for param in layer.parameters():
                                if param.requires_grad:
                                    param.grad = torch.autograd.grad(custom_loss[i], param, retain_graph=True)[0] 
                                    # param.grad = torch.autograd.grad(custom_loss.mean(), param, retain_graph=True)[0] 
                                    
                                    #nn.utils.clip_grad_norm_(self.parameters(), 0.05)
                                    optimizer.step()
                        
        self.eval()     
        
        if self.params['final_grad']:
            loss_total = cum_loss / len(loader) + cum_custom_loss
        else:
            loss_total = cum_loss / len(loader)  + cum_custom_loss / len(loader)
            
        return loss_total, optimizer

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
        # optimizer = opt.Adadelta(
        #     self.parameters(), lr=self.params["lr"], #momentum=self.params["momentum"]
        # )
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
    
    def is_orthogonal_parameter(self, param):
        """
        Check if the given parameter is part of any orthogonal layer in the model.
    
        Parameters:
        - param: The parameter to check.
    
        Returns:
        - True if param is part of any layer.Q, False otherwise.
        """
        return any(param is q_param for layer in self.orthogonal_transform for q_param in layer.parameters())
    


class loss_fun(nn.Module):
    """Loss function."""    

    def forward(self, out, mask=None):
        """forward."""
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()  # pylint: disable=not-callable
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()  # pylint: disable=not-callable

        coagulation_loss = 0.0
        if mask is not None:
            z_mask = out[mask]
            coagulation_loss = (z_mask - z_mask.mean(dim=0)).norm(dim=1).sum()
            
        return -pos_loss - neg_loss + torch.sigmoid(coagulation_loss) - 0.5
            

class ortho_loss(nn.Module):
    """ custom loss based on orthogonal transform distance """
    
    def forward(self, out, data=None, n_id=None, n_batch=None, dist_type='dynamic'):
               
        # extract the directional derivatives per system
        features, indices = geometry.split_by_system(n_id[:n_batch], data.system, out,) #) limit_rows=False)

        # get condition ids for each system        
        cons = [data.condition[n_id[idx]].squeeze() for idx in indices]
        
        # compute distance between each system
        dist = distance(features, cons, dist_type=dist_type,  return_paired=True)
        
        return dist # + pos_dist # + vec2_dist_1 + vec2_dist_2 + vec2_dist_3
        #return vec_dist
        #return dist



    # # procrustes requires us to have same size matrices
    # if limit_rows:
    #     max_rows = min([u.shape[0] for u in dds])
    #     return [dd[:max_rows,:] for dd in dds], indices
    # else:
    #     return dds, indices

def euclidean_distance(a, b):
    return torch.norm(a - b, dim=1)  # Compute Euclidean distance


def distance(embeddings, condition, dist_type='dynamic', return_paired=False):    
    distances = torch.zeros([len(embeddings),len(embeddings)])
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):  # Avoid redundant calculations
            if dist_type=='mmd':
                dist = mmd_distance(embeddings[i], embeddings[j])
            if dist_type=='positional':  
                dist = positional_distance(embeddings[i], embeddings[j])
            if dist_type=='dynamic':
                dist = dynamic_distance(embeddings[i], embeddings[j])
            if dist_type=='condition_procrustes':                
                dist = positional_distance(embeddings[i], embeddings[j])
                intersection = utils.torch_intersect(condition[i].unique(), condition[j].unique())
                condition_emb_1 = [embeddings[i][condition[i]==c,:] for c in intersection]
                condition_emb_2 = [embeddings[j][condition[j]==c,:] for c in intersection]
                dist_condition = condition_distance(condition_emb_1,condition_emb_2)
                dist = dist + dist_condition

            distances[i,j] = dist
            
    # symmetric matrix of distances
    distances = distances + distances.T
    
    # compute sum of distances ignoring the diagonal zeros
    # distance_sum = (distances * torch.ones_like(distances, dtype=torch.bool)).mean(dim=1)
    distance_sum = distances #[:,0]
    
    if return_paired:
        return distance_sum
    else:
        return distance_sum.mean()
    

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1. / (2. * sigma ** 2)
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-beta * dist)

def mmd_distance(x, y, sigma=1.0):
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

def dynamic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes singular value decomposition of product of vectors
    """

    x = nn.functional.normalize(x,p=2,dim=1)
    y = nn.functional.normalize(y,p=2,dim=1)
    
    #min_shape = min(x.shape[0],y.shape[0])
    #x = x[:min_shape,:]
    #y = y[:min_shape,:]

    # Compute SVD of the product of X and Y.T
    dot = x @ y.T
    #dot = torch.pow(dot,2) * torch.sign(dot)
    s = torch.sum(dot) / (dot.shape[0] * dot.shape[1])
    s = (s + 1)/2 # zero is minimum and 1 is max
    return 1-s

def positional_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes singular value decomposition of product of vectors
    """

    x = _matrix_normalize(x, dim=0)
    y = _matrix_normalize(y, dim=0)

    # Compute SVD of the product of X and Y.T
    u, s, v = torch.linalg.svd(x @ y.T)  # or x @ y.T
    s = torch.sum(s) 
    return 1-s


def plot3d(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x[:,0].cpu().detach().numpy(),
                x[:,1].cpu().detach().numpy(),
                x[:,2].cpu().detach().numpy(), c = 'b', marker='o')
    ax.scatter(y[:,0].cpu().detach().numpy(),
                y[:,1].cpu().detach().numpy(),
                y[:,2].cpu().detach().numpy(), c = 'r', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

def plot2d(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111,)
    
    ax.scatter(x[:,0].cpu().detach().numpy(),
                x[:,1].cpu().detach().numpy(), c = 'b', marker='o')
    ax.scatter(y[:,0].cpu().detach().numpy(),
                y[:,1].cpu().detach().numpy(), c = 'r', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    

    

def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    # Step 1: Normalize the vectors in matrices x and y
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)

    # Step 2: Compute the dot product between each pair of vectors
    # This requires expanding dimensions to perform a 'broadcasted' dot product
    dot_product = torch.matmul(x_norm[:, None, :], y_norm.transpose(0, 1).unsqueeze(0))

    # The result of dot_product is a (n, 1, m) tensor. Squeeze it to (n, m) for ease of use
    dot_product = dot_product.squeeze()

    # Step 3: Calculate the angle (in radians) between each pair of vectors
    angles = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # Clamp values to avoid numerical errors

    # Step 4: Compute the average angle
    average_angle = torch.mean(angles)

    return average_angle



def hausdorff_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    x = _matrix_normalize(x, dim=0)
    y = _matrix_normalize(y, dim=0)
    
    # Compute pairwise distances
    # Expanding set1 and set2 into 3D tensors for broadcasting
    d_matrix = torch.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=2))
    
    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the values
    ax.scatter(x[:,0].cpu().detach().numpy(),
                x[:,1].cpu().detach().numpy(),
                x[:,2].cpu().detach().numpy(), c = 'b', marker='o')
    ax.scatter(y[:,0].cpu().detach().numpy(),
                y[:,1].cpu().detach().numpy(),
                y[:,2].cpu().detach().numpy(), c = 'r', marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # plt.show()

    
    # Compute the Hausdorff distance
    min_dist_1, _ = torch.min(d_matrix, dim=1)
    min_dist_2, _ = torch.min(d_matrix, dim=0)
    hausdorff_dist = torch.max(torch.max(min_dist_1), torch.max(min_dist_2))
    
    return hausdorff_dist


def condition_distance(conditions_x, conditions_y):
    """
    Computes euclidean distance between conditions

    :param conditions_x: List of torch.Tensors, each representing a condition of system 1
    :param conditions_y: List of torch.Tensors, each representing a condition of system 2
    :return: Procrustes distance
    """
    assert len(conditions_x) == len(conditions_y), "Both systems must have the same number of conditions"

    distances = torch.zeros(len(conditions_x))

    for i, (x, y) in enumerate(zip(conditions_x, conditions_y)):        
        dist = torch.cdist(x, y).mean()
        distances[i] = dist    
    
    # Aggregate distances across conditions
    total_distance = sum(distances) / len(distances)
    
    #print(total_distance)
    return total_distance

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



