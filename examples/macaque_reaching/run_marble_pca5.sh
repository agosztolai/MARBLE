#!/bin/bash
#BATCH --job-name MARBLE # Name for your job
#SBATCH --ntasks 2             # Number of (cpu) tasks
#SBATCH --time 1200             # Runtime in minutes.
#SBATCH --mem 10000             # Reserve 10 GB RAM for the job
#SBATCH --partition gpu         # Partition to submit ('gpu' or 'cpu')
#SBATCH --qos students             # QOS ('staff' or 'students')
#SBATCH --output myjob-%j.txt       # Standard out goes to this file
#SBATCH --error myjob-%j.txt        # Standard err goes to this file
#SBATCH --mail-user adam.gosztolai@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage, can be 'teslak40', 'gtx1080', or 'titanrtx'
#SBATCH --chdir /mnt/scratch/lts2/gosztolai/MARBLE/examples/macaque_reaching

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate MARBLE

# RUN TRAINING
python -m run_marble_pca5
