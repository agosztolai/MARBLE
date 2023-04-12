import numpy as np
import sys
import neo

from quantities import ms
from scipy.io import loadmat
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    """Extracting kinematic data from matlab into format for decoding"""

    folder = "/media/robert/Extreme SSD/ResearchProjects/MARBLE/lfads-neural-stitching-reproduce/results/"
    file = "kinematics_lfadsSingleFromFactors.mat"
    kinematic_data = loadmat(folder + file)["Tin_single"]

    trial_ids = pickle.load(open("../data/trial_ids.pkl", "rb"))

    kinematics = {}
    conditions = ["DownLeft", "Left", "UpLeft", "Up", "UpRight", "Right", "DownRight"]

    plot = False

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

            # kinematics_conds[cond].append(X)
            # lfads_factors_conds[cond].append(Z)

        kinematics[d] = kinematics_conds
        # lfads_factors[d] = lfads_factors_conds

        if plot:
            plt.figure()
            for cond in conditions:
                meh = np.dstack(kinematics_conds[cond]).mean(2)
                plt.plot(meh[0, :], meh[1, :])

    with open("../data/kinematics.pkl", "wb") as handle:
        pickle.dump(kinematics, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_condition(val, trials, conditions):
    for cond in conditions:
        if val in trials[cond]:
            return cond


if __name__ == "__main__":
    sys.exit(main())
