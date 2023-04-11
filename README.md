# MARBLE - **Ma**nifold **R**epresentation **B**asis **L**earning

This package contains a geometric deep learning method to intrincally represent vector and scalar fields over manifolds and compare representations obtained from different vector fields. The examples  

The code is built around [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).\

## Cite

If you find our work useful or inspirational, please cite our work as follows

```
@inproceedings{GosztolaiGunel20LiftPose3D,
  author    = {Adam Gosztolai and
               Semih Günel and
               Marco Pietro Abrate and
               Daniel Morales and 
               Victor Lobato Rios and
               Helge Rhodin and
               PascalFua and
               Pavan Ramdya},
  title     = {LiftPose3D, a deep learning-based approach for transforming 2D to 3D pose in laboratory experiments},
  bookTitle = {bioRxiv},
  year      = {2020}
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
