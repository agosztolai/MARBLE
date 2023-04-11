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

Comparing dynamics in a data-driven way across simulations or experiments then becomes equivalent to comparing the corresponding vector fields over distinct manifolds (!) based on their respective sample sets. 



```
x_list, v_list = [], []
```

and them
data = MARBLE.construct_dataset(pos_concat, features=vel_concat, graph_type='cknn', k=15, stop_crit=0.03, vector=False)

Then, you obtain vectors `v1 = numpy.diff(x1, axis=0)` and likewise for `x2`, and finally stack them vertically `x = np.vstack([x1, x2])` 

x = 
v = 


## Examples

The folder `/examples` contains some example scripts.

## References

The following packages were inspirational during the delopment of this code:

* [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
* [Directional Graph Networks](https://github.com/Saro00/DGN)
* [pyEDM](https://github.com/SugiharaLab/pyEDM)
* [Parallel Transport Unfolding](https://github.com/mbudnins/parallel_transport_unfolding)
