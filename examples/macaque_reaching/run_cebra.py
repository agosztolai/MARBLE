import pickle
import os
import sys
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from run_marble import load_data, fit_pca, format_data
from tqdm import tqdm
from cebra import CEBRA


def main():

    """fitting model for each day + pca embedding"""
    
    data_file = "data/rate_data_20ms.pkl"
    metadata_file = "data/trial_ids.pkl"
    
    rates, trial_ids = load_data(data_file, metadata_file)

    # defining the set of conditions
    conditions = ["DownLeft", "Left", "UpLeft", "Up", "UpRight", "Right", "DownRight"]

    # list of days
    days = rates.keys()

    # define some parameters
    pca_n = 5
    filter_data = True

    # storing all distance matrices
    embeddings = []
    distance_matrices = []
    times = [] # to store the time point of each node in the trajectory
    all_condition_labels = [] # to store the condition label for each node
    all_trial_ids = [] # trial ids for each node
    all_sampled_ids = [] # to store all the nodes sampled by marble
    
    # loop over each day
    for day in tqdm(days):

        # first stack all trials from that day together and fit pca
        print(day)
        pca = fit_pca(rates, day, conditions, filter_data=filter_data, pca_n=pca_n)
        pos, vel, timepoints, condition_labels, trial_indexes = format_data(rates, 
                                                                            trial_ids,
                                                                            day, 
                                                                            conditions, 
                                                                            pca=pca,
                                                                            filter_data=filter_data)
            
        
        cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.0001,
                        temperature=1,
                        output_dimension=20,
                        max_iterations=5000,
                        distance='euclidean',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

        pos_all = np.vstack(pos)
        condition_labels = np.hstack(condition_labels)
        cebra_model.fit(pos_all, condition_labels)
        cebra_pos = cebra_model.transform(pos_all)

        cebra_model.save("data/session_{}_20ms.pt".format(day))

        embeddings.append(cebra_pos)
        distance_matrices.append([])
        times.append(np.hstack(timepoints))
        all_condition_labels.append(np.hstack(condition_labels))
        all_trial_ids.append(np.hstack(trial_indexes))
        all_sampled_ids.append([])

        # save over after each session (incase computations crash)
        with open("data/cebra_embeddings_20ms_out20.pkl", "wb") as handle:
            pickle.dump(
                [
                    distance_matrices,
                    embeddings,
                    times,
                    all_condition_labels,
                    all_trial_ids,
                    all_sampled_ids,
                ],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    # final save
    with open("data/cebra_embeddings_20ms_out20.pkl", "wb") as handle:
        pickle.dump(
            [
                distance_matrices,
                embeddings,
                times,
                all_condition_labels,
                all_trial_ids,
                all_sampled_ids,
            ],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

if __name__ == "__main__":
    sys.exit(main())
