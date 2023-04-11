# MARBLE - **Ma**nifold **R**epresentation **B**asis **L**earning

This package contains a geometric deep learning method to intrincally represent vector and scalar fields over manifolds and compare representations obtained from different vector fields. The examples  

The code is built around [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Cite

If you find our work useful or inspirational, please cite our work as follows

```
@misc{gosztolai2023interpretable,
      title={Interpretable statistical representations of neural population dynamics and geometry}, 
      author={Adam Gosztolai and Robert L. Peach and Alexis Arnaudon and Mauricio Barahona and Pierre Vandergheynst},
      year={2023},
      eprint={2304.03376},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Installation

We recommend you install the code on a fresh Anaconda virtual environment, as follows.

First, clone this repository, 

```
git clone https://github.com/agosztolai/GeoDySys
```

Then, create an new anaconda environment using the provided environment file that matches your system.

For Linux machines with CUDA: 

```
conda env create -f environment.yml
```

For Mac without CUDA:

```
conda env create -f environment_cpu_osx.yml
```


This will install all the requires dependencies. Finally, install by running inside the main folder

```
pip install . 
```

## Workflow

### Input

Our method takes as inputs an `nxd` vector `x`, which consists of n vectors defining the manifold shape, and a corresponding `nxd` vector `v`, which consists of vectors defining the dynamics over the manifold. 

If you measure time series observables, such as neural firing rates, you can start with a list of variable length trajectories which make up the data of a dynamical system under a given condition `x = list(time series 1, time series 2 ...)`. 

If you only measure `x`, you can also get the velocities as `v = [np.diff(x, axis=0) for x_ in x]` and take `x = [x_[:-1,:] for x_ in x]` to ensure they have the same length. 

We assume that the trajectories under a given condition are settled over the same manifold. Therefore, we consider the `x`, `y` pairs as samples of the vector field over the manifold, and stack them row-wise.

```
x, v = np.vstack(x), np.vstack(v)
```

### Handling different conditions, or dynamical systems

Given the above setup, comparing dynamics in a data-driven way across simulations or experiments becomes equivalent to comparing the corresponding vector fields based on their respective sample sets. Different simulations may correspond to different experimental conditions, measurements, dynamical systems between which we seek to find similarities.

Suppose we have the data pairs `x1, v1` and `x2, v2`. Then concatenating them as a list means they arise as samples from distinct manifolds and will be handled independently by our pipeline. 

```
x_list, v_list = [x1, x2], [v1, v2]
```

It is also sometimes useful to consider that two vector fields lie on independent manifolds when we want to discover the contrary. However, when we know that two vector fields lie on the same manifold, then it can be advantageous to stack their corresponding samples, rather than providing them as a list, as this will enforce geometric relationships between them.

### Constructing data object

Our pipleline is build around a Pytorch Geometric data object, which we can obtain by running the following constructor.

```
import MARBLE 

data = MARBLE.construct_dataset(x_list, features=v_list, graph_type='cknn', k=15, stop_crit=0.03, curvature=False)
```

This command will first subsample each point cloud using farthest point sampling to achive even sampling. Using `stop_crit=0.03` means the average distance between the subsampled points will equal to 3% of the manifold diameter. Then it will fit a nearest neighbour graph to each point cloud, here using the `cknn` method using `k=15` nearest neighbours. 

The final argument, `curvature=False` means that local patches will be treated as flat Euclidean spaces. Setting `curvature=True` can be useful when the sampling is sparse relative to the manifold curvature, however, this will increase the cost of the computations.

The final data object contains the following attributes:

```
data.pos: positions x
data.x: vectors v
data.y: labels for each points denoting which manifold it belongs to
```

### Training

You are ready to train! This is straightforward.

You first specify the hyperparameters. The key ones are the following (see `/MARBLE/default_params.yaml` for a complete list), which will work for many settings.

```
params = {'epochs': 50, #optimisation epochs
          'order': 2, #order of derivatives
          'hidden_channels': 32, #number of internal dimensions in MLP
          'out_channels': 5,
          'inner_product_features': True,
         }

```

Then proceed by constructing a network object

```
model = MARBLE.net(data, params=params) #loadpath='model_large' )
```

Finally, launch training. The code will continuously save checkpoints during training with timestamps. 

```
model.run_training(data, outdir='./outputs')
```

If you have previously trained a network, you can skip the training step and load the network directly as

```
model = MARBLE.net(data, loadpath=loadpath, params=params)
```

where loadpath can be either a path to the model (with a specific timestamp, or a directory to automatically load the last model.

### Evaluating the network

A given network can be evaluated on the data on which it was trained on or another dataset (currently not tested). The following line appends a `data.emb` attribute, which is an `nxout_channels` matrix containing the embeddings for individual sample points in all datasets. 

```
data = model.evaluate(data)
```

To recover the identity of the datasets, use `data.y`, e.g., `data.emb[data.y==0]` to obtain the embedding for the vector field over the first manifold.

We can also run postprocessing, which computes the distributional distances between the datasets.

```
data = postprocessing(data)
```

Now we should have a `data.dist` attribute corresponding to a symmetric distance matrix. We can also use an optional `n_clusters` argument to apply a k-means kernel to the embedded points before computing distance, which we found to improve performance.

## Examples

The folder `/examples` contains scripts for some basic examples and reproduction.

To get more intuition of MARBLE, we suggest you start by running `ex_vector_field_flat_surface.py` or `ex_vector_field_curved_surface.py`.


## References

The following packages were inspirational during the delopment of this code:

* [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
* [Directional Graph Networks](https://github.com/Saro00/DGN)
* [pyEDM](https://github.com/SugiharaLab/pyEDM)
* [Parallel Transport Unfolding](https://github.com/mbudnins/parallel_transport_unfolding)
