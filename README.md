# MARBLE - **Ma**nifold **R**epresentation **B**asis **L**earning

This package contains a geometric deep learning method to intrincally represent vector and scalar fields over manifolds and compare representations obtained from different vector fields. The examples  

The code is built around [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).\

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

Using our code follows

## Examples

The folder `/examples` contains some example scripts.

## References

The following packages were inspirational during the delopment of this code:

* [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
* [Directional Graph Networks](https://github.com/Saro00/DGN)
* [pyEDM](https://github.com/SugiharaLab/pyEDM)
* [Parallel Transport Unfolding](https://github.com/mbudnins/parallel_transport_unfolding)
