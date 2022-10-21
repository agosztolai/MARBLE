# MARBLE - **Ma**nifold **R**epresentation **B**asis **L**earning

This package contains a geometric deep learning method to intrincally represent vector and scalar fields over manifolds and compare representations obtained from different vector fields. The examples  

The code is built around [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Getting started

We recommend you install the code on a fresh Anaconda virtual environment, as follows.

First, clone this repository, 

```
git clone https://github.com/agosztolai/GeoDySys
```

Then, create an new anaconda environment using the provided environment.yaml file,

```
conda env create -f environment.yml
```

This will install all the requires dependencies. Finally, install by running inside the main folder

```
pip install . 
```

## Examples

The folder `/examples` contains some example scripts.

## References

The following packages were inspirational during the delopment of this code:

* [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
* [Directional Graph Networks](https://github.com/Saro00/DGN)
* [pyEDM](https://github.com/SugiharaLab/pyEDM)
* [Parallel Transport Unfolting](https://github.com/mbudnins/parallel_transport_unfolding)
