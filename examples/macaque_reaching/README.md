# Spiking data example

This folder contains scripts to generate the results presented in Figure 3 in the paper.

The scripts and notebooks are as follows. 

1. If you just want to reproduce the results, the following notebooks will automatically download precomputed MARBLE embeddings from Harvard Dataverse

- plot_vector_fields.ipynb: this notebook will plot the firing rate data as vector fields (Fig. 4d)
- plot_MARBLE_representations.ipynb: this notebook will plot MARBLE representations (Fig. 4e)
- kinematic_decoding.ipynb: this notebook will plot the kinematic decoding results (Fig. 4f,g)

2. If you want to rerun from spiking neuron data

- convert_spikes_to_firing_rates.py: convert spike train data into firing rate data that is compatible with MARBLE
- run_marble.py: train MARBLE networks (this takes a while to run - 1-2h/session)
- run_cebra.py: train CEBRA networks (this takes a while to run - 1-2h/session)