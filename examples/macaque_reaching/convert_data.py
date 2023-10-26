"""Convert spiking and kinematics data into instantaneous rates for MARBLE analysis.

It needs neo and elephant packages in addition to MARBLE.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from elephant.statistics import instantaneous_rate
import neo
from elephant.kernels import GaussianKernel
from quantities import ms
from MARBLE import utils
import mat73
import pickle


def spikes_to_rates(data, d, sampling_period=20):
    """
    Converts matlab spiking data into instantaneous rates in a suitable format for further analysis
    """

    # defining conditions by their ordering (this was how it was ordered in matlab script)
    conditions = ["DownLeft", "Left", "UpLeft", "Up", "UpRight", "Right", "DownRight"]

    data_day = data[d]  # daily session

    gk = GaussianKernel(100 * ms)  # increase this for smoother signals (previously used auto)

    # define empty dictionary for each day
    rates = {}

    # loop over the 7 conditions
    for c, cond in enumerate(conditions):

        # define empty list for each condition during each session
        trial_data = []

        # extracting data for a single condition on a single day (composed of t trials)
        data_day_cond = data_day[c]

        # loop over trials
        for t, trial in enumerate(data_day_cond):

            # if the trial exists (sometimes there is None)
            if trial:
                trial = trial[0]  # it was a single element list

                # loop over neurons
                inst_rates = []
                for ch in range(trial.shape[0]):

                    # extract spikes for a given channel (neuron)
                    spikes = np.where(trial[ch, :])[0]

                    # get spike train (1200 milliseconds)
                    st = neo.SpikeTrain(spikes, units="ms", t_stop=1200)

                    # get rates
                    inst_rate = instantaneous_rate(st, kernel=gk, sampling_period=sampling_period*ms).magnitude

                    # append into list
                    inst_rates.append(inst_rate.flatten())

                # stack rates back together and transpose = (channels by time)
                inst_rates = np.stack(inst_rates, axis=1)

                # append rates from one trial into trial data
                trial_data.append(inst_rates)

        # stack into an array of trial x channels x time
        rates[cond] = np.dstack(trial_data).transpose(2, 0, 1)

    return rates


def convert_spiking_rates(sampling_period=20):

    data_file = "data/conditions_spiking_data.mat"
    Path("data").mkdir(exist_ok=True)
    os.system(f"wget -nc https://dataverse.harvard.edu/api/access/datafile/6963157 -O {data_file}")

    # load data compiled into matlab cell array
    data = mat73.loadmat(data_file)["all_results"]

    rates = utils.parallel_proc(
        spikes_to_rates, range(len(data)), data, processes=-1, desc="Converting spikes to rates..."
    )

    all_rates = {}
    for i, rates_day in enumerate(rates):
        all_rates[i] = rates_day

    with open(f"data/rate_data_{sampling_period}ms.pkl", "wb") as handle:
        pickle.dump(all_rates, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_condition(val, trials, conditions):
    for cond in conditions:
        if val in trials[cond]:
            return cond


def convert_kinematics():
    """Extracting kinematic data from matlab into format for decoding"""

    data_file = "data/kinematics_lfadsSingleFromFactors.mat"
    os.system(f"wget -nc  https://dataverse.harvard.edu/api/access/datafile/7062085 -O {data_file}")
    kinematic_data = loadmat(data_file)["Tin_single"]

    data_file = "data/trial_ids.pkl"
    os.system(f"wget -nc  https://dataverse.harvard.edu/api/access/datafile/6963200  -O {data_file}")    
    trial_ids = pickle.load(open("./data/trial_ids.pkl", "rb"))

    kinematics = {}
    conditions = ["DownLeft", "Left", "UpLeft", "Up", "UpRight", "Right", "DownRight"]

    for d, day in enumerate(kinematic_data):
        day = day[0]
        kinematics_conds = defaultdict(list)

        for i in range(day.shape[0]):
            X = day[i].tolist()[0][1]  # kinematics x,y posiution and x,y velocity
            Z = day[i].tolist()[0][2]  # lfads factors
            T = day[i].tolist()[0][3]  # time

            cond = find_condition(i, trial_ids[d], conditions)

            kinematics_conds[i] = {
                "kinematics": X,
                "lfads_factors": Z,
                "time": T,
                "condition": cond,
            }

        kinematics[d] = kinematics_conds

    with open("data/kinematics.pkl", "wb") as handle:
        pickle.dump(kinematics, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    #convert_spiking_rates()
    convert_kinematics()


if __name__ == "__main__":
    sys.exit(main())
