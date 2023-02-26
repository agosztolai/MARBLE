#!/bin/bash
#SBATCH --job-name spiking_to_rates # Name for your job
#SBATCH --ntasks 16              # Number of (cpu) tasks
#SBATCH --time 120             # Runtime in minutes.
#SBATCH --mem 10000             # Reserve 10 GB RAM for the job
#SBATCH --partition cpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos staff             # QOS ('staff' or 'students')
#SBATCH --output myjob-%j.txt       # Standard out goes to this file
#SBATCH --error myjob-%j.txt        # Standard err goes to this file
#SBATCH --mail-user adam.gosztolai@epfl.ch
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
           # Reserve 1 GPU for usage, can be 'teslak40', 'gtx1080', or 'titanrtx'
#SBATCH --chdir . 

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate MARBLE

# RUN TRAINING
python spiking_to_rates.py
