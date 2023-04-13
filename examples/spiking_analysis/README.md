# Spiking data example

This folder contains scripts to generate the results presented in Figure 3 in the paper.

The scripts and notebooks are as follows. 

1. If you just want to reproduce the results

- rate_data_visualisation.ipynb: create plots of some panels of Figure 3

2. If you want to rerun from spiking neuron data

- convert_data.py: convert original spiking and kinematics data to data compatible with MARBLE (both input and generated data are on dataverse)
- per_session_analysis_20ms.py: train MARBLE networks (this takes a while to run - 1-2h/session)
- rate_data_visualisation.ipynb: create plots of some panels of Figure 3

3. Decoding into kinematics
- decoding_per_session_20ms.ipynb
