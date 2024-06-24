# MARBLE - MAnifold Representation Basis LEarning

MARBLE is an unsupervised geometric deep-learning method that can 

1. Find interpretable latent representations of neural dynamics. It also applies to non-linear dynamical systems in other domains or, more generally, vector fields over manifolds.
2. Perform unbiased comparisons across conditions within the same animal (or dynamical system).
3. Compare dynamics across animals or artificial neural networks.

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

## Documentation

See full documentation [here](https://agosztolai.github.io/MARBLE/).


## Installation

The code is tested for CPU and GPU (CUDA) machines running Linux, Mac OSX or Windows. Although smaller examples run fast on CPU, for larger datasets, it is highly recommended that you use a GPU machine.

We recommend you install the code in a fresh Anaconda virtual environment, as follows.

- First, clone this repository, 

```
git clone https://github.com/agosztolai/MARBLE
```

- Then, create a new Anaconda environment using the provided environment file that matches your system.
  - For Linux machines with CUDA:

  `conda env create -f environment.yml`
  - For Intel-based Mac:

  `conda env create -f environment_osx_intel.yml`

  - For recent M1/M2/M3 Mac:
    - Install cmake `brew install cmake` or use the installer on the [cmake website](https://cmake.org/download/)
    - Create the environment

        `conda env create -f environment_osx_arm.yml`
    - Activate the environment `conda activate MARBLE` 
    - Install PyTorch geometric
    `pip install -r requirements_osx_arm.txt`

  - For Windows computers:
  we recommend using WSL2, which allows running a (virtual) Linux machine inside your Windows computer, which makes the installation simpler. If you have a NVIDIA GPU, WSL2 will allow you to take advantage of the GPU (an older version of WSL will not).
    - Follow the [instructions to install WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
    - Open "Ubuntu" and install a compiler `sudo apt update && sudo apt install gcc g++`
    - Proceed with conda install and environment creation as described for Linux machines.
    - If you do not want to use WSL, this is possible, albeit more complicated. You need to have a working compiler (e.g. Visual Studio or [MSYS2](https://www.msys2.org/)). Once installed, along with conda you can create the Python environment using `conda env create -f environment_windows_native.yml`. 
- All the required dependencies are now installed. Finally, activate the environment and install it by running inside the main folder

```
conda activate MARBLE
pip install . 
```

### Note on examples

Running the scripts in the `/examples` folder to reproduce our results will rely on several dependencies that are not core to the MARBLE code. On Macs, run `brew install wget`, which you will need to download the datasets automatically. Further dependencies will install automatically when running the example notebooks.

## Quick start

We suggest you study at least the example of a [simple vector fields over flat surfaces](https://github.com/agosztolai/MARBLE/blob/main/examples/toy_examples/ex_vector_field_flat_surface.py) to understand what behaviour to expect.

Briefly, MARBLE takes two inputs

1. `anchor` - a list of `nxd` arrays, each defining a cloud of anchor points describing the manifold
2. `vector` - a list of `nxD` arrays, defining a vector signal at the anchor points over the respective manifolds. For dynamical systems, D=d, but our code can also handle signals of other dimensions. Read more about [inputs](#inputs) and [different conditions](#conditions).

Using these inputs, you can construct a dataset for MARBLE.

```
import MARBLE 
data = MARBLE.construct_dataset(anchor=pos, vector=x)
```

The main attributes are `data.pos` - manifold positions concatenated, `data.x` - manifold signals concatenated and `data.y` - identifiers that tell you which manifold the point belongs to. Read more about [other useful data attributes](#construct).

Now, you can initialise and train a MARBLE model. Read more about [training parameters](#training).

```
from MARBLE import net
model = MARBLE.net(data)
model.fit(data)
```

By default, MARBLE operates in embedding-aware mode. To enable the embedding-agnostic mode, change the initialisation to

```
model = MARBLE.net(data, params = {'inner_product_features': True})
```

Read more about the embedding-aware and embedding-agnostic modes [here](#innerproduct)

After you have trained your model, you can evaluate your model on your dataset or another dataset to obtain an embedding of all manifold points in joint latent space (3-dimensional by default) based on their local vector field features.

```
data = model.transform(data) #adds an attribute `data.emb`
```

To recover the embeddings of individual vector fields, use `data.emb[data.y==0]`, which takes the embedding of the first vector field.

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

There are loads of parameters to adjust these plots, so look at the respective functions.

## Examples

The folder [/examples](https://github.com/agosztolai/MARBLE/tree/main/examples) contains scripts for some basic examples and other scripts to reproduce the results in our paper.

## Further details

<a name="inputs"></a>
### More on inputs

If you measure time series observables, such as neural firing rates, you can start with a list of variable-length time series under a given condition, e.g., `ts_1`, `ts_2`. We assume these are measurements from the same dynamical system, i.e., the sample points making up these trajectories are drawn from the same manifold, defining its shape `pos = np.vstack([ts_1, ts_2])`.

If you do not directly have access to the velocities, you can approximate them as `x = np.vstack([np.diff(ts_1, axis=0), np.diff(ts_2, axis=0)])` and take `pos = np.vstack([ts_1[:-1,:], ts_2[:-1,:]])` to ensure `pos` and `x` have the same length. 

If you just want to play around with dynamical systems, why not try our (experimental) sister package [DE_library](https://github.com/agosztolai/DE_library).

<a name="conditions"></a>
### More on different conditions

Comparing dynamics in a data-driven way is equivalent to comparing the corresponding vector fields based on their respective sample sets. The dynamics to be compared might correspond to different experimental conditions (stimulation conditions, genetic perturbations, etc.) and dynamical systems (different tasks, different brain regions).

Suppose we have the data pairs `pos1, pos2` and `x1, x2`. Then we may concatenate them as a list to ensure that our pipeline handles them independently (on different manifolds), but embeds them jointly in the same space.

```
pos_list, x_list = [pos1, pos2], [x1, x2]
```

It is sometimes useful to consider that two vector fields lie on independent manifolds (providing them as a list) even when we want to *discover* the contrary. However, when we know that two vector fields lie on the same manifold, then it can be advantageous to stack their corresponding samples (stacking them) as this will enforce geometric relationships between them through the proximity graph.

<a name="construct"></a>
### More on constructing data object

Our pipeline is built around a Pytorch Geometric data object, which we can obtain by running the following constructor.

```
import MARBLE 
data = MARBLE.construct_dataset(anchor=pos, vector=x, spacing=0.03, graph_type='cknn', k=15, local_gauge=False)
```

This command will do several things.

1. Subsample the point cloud using farthest point sampling to achieve even sampling density. Using `spacing=0.03` means the average distance between the subsampled points will equal 3% of the manifold diameter.
2. Fit a nearest neighbour graph to each point cloud using the `graph_type=cknn` method using `k=15` nearest neighbours. We implemented other graph algorithms, but cknn typically works. Note that `k` should be large enough to approximate the tangent space but small enough not to connect (geodesically) distant points of the manifold. The more data you have, the higher `k` you can use.
3. Perform operations in local (manifold) gauges or global coordinates. Note that `local_gauge=False` should be used whenever the manifold has negligible curvature on the scale of the local feature. Setting `local_gauge=True` means that the code performs tangent space alignments before computing gradients. However, this will increase the cost of the computations $m^2$-fold, where $m$ is the manifold dimension because points will be treated as vector spaces. See the example of a [simple vector fields over curved surfaces](https://github.com/agosztolai/MARBLE/blob/main/examples/toy_examples/ex_vector_field_curved_surface.py) for illustration.


The final data object contains the following attributes (among others):

```
data.pos: positions `pos` concatenated across manifolds
data.x: vectors `x` concatenated across manifolds
data.y: labels for each point denoting which manifold it belongs to
data.edge_index: edge list of proximity graph (each manifold gets its graph, disconnected from others)
```

<a name="training"></a>
### Training

You are ready to train! This is straightforward.

You first specify the hyperparameters. The key ones are the following, which will work for many settings, but see [here](https://github.com/agosztolai/MARBLE/blob/main/MARBLE/default_params.yaml) for a complete list.

```
params = {'epochs': 50, #optimisation epochs
          'hidden_channels': 32, #number of internal dimensions in MLP
          'out_channels': 5,
          'inner_product_features': True,
         }

```

Then, proceed by constructing a network object

```
model = MARBLE.net(data, params=params)
```

Finally, launch training. The code will continuously save checkpoints during training with timestamps. 

```
model.fit(data, outdir='./outputs')
```

If you have previously trained a network or have interrupted training, you can load the network directly as

```
model = MARBLE.net(data, loadpath=loadpath)
```

where loadpath can be either a path to the model (with a specific timestamp, or a directory to load the latest model automatically. By running `MARBLE.fit()`, training will resume from the last checkpoint.

<a name="innerproduct"></a>
### Embedding-aware and embedding-agnostic modes

One of the main features of our method is the ability to run in two different modes

1. Embedding-aware mode - learn manifold embedding and dynamics
2. Embedding-agnostic mode - learn dynamics only

To enable embedding-agnostic mode, set `inner_product_features=True` in training `params`. This engages an additional layer in the network after the computation of gradients, which makes them rotation invariant.

As a slight cost of expressivity, this feature enables the orientation- and embedding-independent representation of dynamics over the manifolds. Amongst others, this allows one to recognise similar dynamics across different manifolds. See [RNN example](https://github.com/agosztolai/MARBLE/blob/main/examples/RNN/RNN.ipynb) for an illustration.


## Troubleshooting guide

Training is successful when features are recognised to be similar across distinct vector fields with their own manifolds and independent proximity graphs. To achieve this, follow these useful pieces of advice (mostly general ML practises):

1. Check that training has converged, i.e., the validation loss is no longer decreasing.
2. Check that convergence is smooth, i.e., there are no big jumps in the validation loss.
3. Check that there is no big gap between training loss and validation loss (generalisation gap). 

Problems with the above would be possible signs your solution will be suboptimal and will likely not generalise well. In this case, try the following
 * increase training time (increase `epochs`)
 * increase your data (e.g., decrease `spacing` and construct the dataset again)
 * decrease number of parameters (decrease `hidden_channels`, or decrease order, try `order=1`)
 * improve the gradient approximation (increase `k` or `delta`)

If your data is very noisy, try enabling diffusion (`diffusion=True` in training `params`).

If this still does not work, check for very small or very large vector magnitudes in your dataset, filter them out, and try again. 


## Stay in touch

If all hope is lost, or if you want to chat about your use case, get in touch or raise an issue! We are happy to help and looking to further develop this package to make it as useful as possible.


## References

The following packages were inspirational during the development of this code:

* [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
* [Directional Graph Networks](https://github.com/Saro00/DGN)
* [pyEDM](https://github.com/SugiharaLab/pyEDM)
* [Parallel Transport Unfolding](https://github.com/mbudnins/parallel_transport_unfolding)
