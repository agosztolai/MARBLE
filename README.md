# MARBLE - **Ma**nifold **R**epresentation **B**asis **L**earning

MARBLE is a fully unsupervised geometric deep learning method that can 

1. intrincally represent vector fields over manifolds, such as those arising in neural recordings or dissipative dynamical systems, but it is not limited to dynamical systems
2. perform unbiased comparisons across dynamical systems or parameter conditions by jointly embedding representations
3. can operate in geometry-aware of geometry-agnostic mode to test the contribution of manifold dynamics and geometry.

The code is built around [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Cite

If you find this package useful or inspirational, please cite our work as follows

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

## Quick start

We suggest you study at least the scripts of a [simple vector fields over flat surfaces](https://github.com/agosztolai/MARBLE/blob/main/examples/ex_vector_field_flat_surface.py) and [simple vector fields over curved surfaces](https://github.com/agosztolai/MARBLE/blob/main/examples/ex_vector_field_curved_surface.py) to understand what behaviour to expect.

Briefly, MARBLE takes two inputs

1. `pos` - a list of `nxd` arrays, each defining a point cloud describing the geometry of a manifold
2. `vel` - a list of `nxD` arrays, defining a signal over the respective manifolds in 1. For vector fields D=d, but our code can also handle signals of other dimensions. Read more about [inputs](#inputs) and [different conditions](#conditions).

Then construct a dataset for MARBLE.

```
import MARBLE 
data = MARBLE.construct_dataset(pos, features=vel)
```

The main attributes are `data.pos` - manifold positions concatenated, `data.x` - manifold signals concatenated and `data.y` - identifiers that tell you which manifold the poitn belongs to. Read more about [other usedul data attributed](#construct).

Now you can initialise and train a MARBLE model. 

```
from MARBLE import net
model = MARBLE.net(data)
model.run_training(data)
```

By default, MARBLE operated in geometry-aware mode. You can enable the geometry-agnostic mode by changing the initalisation step to

```
model = MARBLE.net(data, params = {'inner_product_features': True})
```

After you have trained your model, you can evaluate evaluate your model on your dataset, or another dataset to obtain an embedding all manifold points in joint latent space (3-dimensional by default) based on their local vector field features.

```
data = model.evaluate(data) #adds an attribute `data.emb`
```

To recover the embeddings of individual vector fields, use `data.emb[data.y==0]`.

You can then compare datasets by running

```
from MARBLE import postprocessing
data = postprocessing.distribution_distances(data) #adds an attribute `data.dist` containing a matrix of pairwise distance between vector field representations
```

Finally, you can perform some visualisation

```
from MARBLE import plotting
data = postprocessing.embed_in_2D(data) #adds an attribute `data.emb_2D` containing a 2D embedding of the MARBLE output using UMAP by default
plotting.fields(data) #visualise the original vector fields over manifolds 
plotting.embedding(data, data.y.numpy()) #visualise embedding
```

There are loads of parameters to adjust these plots, so have a look at the respective functions.

<a name="inputs"></a>
### More on inputs

If you measure time series observables, such as neural firing rates, you can start with a list of variable length time series under a given condition, e.g., `ts_1`, `ts_2`. We assume these are measurements from the same dynamical system, i.e., the sample points making up these trajectories are drawn from the same manifold, defining its shape `pos = np.vstack([ts_1, ts_2])`.

If you do not directly have access to the velocities, you can approximate them as `vel = np.vstack([np.diff(ts_1, axis=0), np.diff(ts_2, axis=0)])` and take `pos = np.vstack([ts_1[:-1,:], ts_2[:-1,:]])` to ensure `pos` and `vel` have the same length. 

<a name="conditions"></a>
### More on different conditions

Comparing dynamics in a data-driven way is equivalent to comparing the corresponding vector fields based on their respective sample sets. The dynamics to be compared might correspond to different experimental conditions (stimulation conditions, genetic perturbations etc.), dynamical systems (different task, different brain region).

Suppose we have the data pairs `pos1, pos2` and `vel2, vel2`. Then we may concatenate them as a list to ensure that our pipeline handles them independently (on different manifolds), but embeds them jointly in the same space.

```
x_list, v_list = [x1, x2], [v1, v2]
```

Note, it is sometimes useful to consider that two vector fields lie on independent manifolds when we want to discover the contrary. However, when we know that two vector fields lie on the same manifold, then it can be advantageous to stack their corresponding samples, rather than providing them as a list, as this will enforce geometric relationships between them.

<a name="construct"></a>
### More on constructing data object

Our pipleline is build around a Pytorch Geometric data object, which we can obtain by running the following constructor.

```
import MARBLE 
data = MARBLE.construct_dataset(pos, features=vel, graph_type='cknn', k=15, stop_crit=0.03, local_gauge=False)
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
