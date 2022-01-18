# Dynamical Ollivier-Ricci curvature and clustering

This package computes the dynamic Ollivier-Ricci based on Markov diffusion processes for edges of a graph, and uses it cluster the graph.

## Cite

Please cite our paper if you use this code in your own work. To reproduce the results of our paper, run the jupyter notebooks in the folder `examples/paper_results`.

```
@article{gosztolaiArnaudon,
  author    = {Adam Gosztolai and
               Alexis Arnaudon},
  title     = {Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature},
  journal = {Preprint at Researchsquare https://www.researchsquare.com/article/rs-222407/v1}
  doi = {10.21203/rs.3.rs-222407/v1}
  year      = {2021}
}
```

## Getting started

### Installation

To install this package, clone this repository, and run

```
pip install . 
```

To run the code is very simple. The folder `/examples` contains some example scripts.

### Data requirements

Our code can be applied to any graph provided as a networkx object. This can be taken from the examples in the folder `/graph`, or provided by the user. You can generate a host of standard graphs using our [graph library package](https://github.com/agosztolai/graph_library)!

### Compute curvature
If taken from the folder `graph`, the multiscale curvature can be computed by running
```
python run_curvature.py <graph>
```

To plot the results, use
```
python plot_curvature.py <graph>
```

To only run the [classical Ollivier-Ricci curvature](https://www.sciencedirect.com/science/article/pii/S002212360800493X), use
```
python compute_original_OR.py <graph>
```

### Compute clustering

The clustering function requires our [PyGenStability package](https://github.com/ImperialCollegeLondon/PyGenStability), which is a Python wrapper for the generalised Louvain algorithm. 

To run clustering using geometric modularity (modularity on curvature weighted graph without null model), run 
```
python run_clustering.py <graph>
```
then plot the results with
```
python plot_clustering.py <graph>
```
