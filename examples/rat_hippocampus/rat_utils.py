import sys
import numpy as np
import matplotlib.pyplot as plt

from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate
from quantities import ms
import neo

from sklearn.decomposition import PCA
import sklearn

import MARBLE
import cebra

def prepare_marble(spikes, labels, pca=None, pca_n=10, skip=1):
    
    s_interval = 1
    
    gk = GaussianKernel(10 * ms)
    rates = []
    for sp in spikes:
        sp_times = np.where(sp)[0]
        st = neo.SpikeTrain(sp_times, units="ms", t_stop=len(sp))
        r = instantaneous_rate(st, kernel=gk, sampling_period=s_interval * ms).magnitude
        rates.append(r.T)

    rates = np.vstack(rates)

    if pca is None:
        pca =  PCA(n_components=pca_n)
        rates_pca = pca.fit_transform(rates.T)
    else:
        rates_pca = pca.transform(rates.T)
        
    vel_rates_pca = np.diff(rates_pca, axis=0)
    print(pca.explained_variance_ratio_)  

    rates_pca = rates_pca[:-1,:] # skip last

    labels = labels[:rates_pca.shape[0]]
    
    data = MARBLE.construct_dataset(
        rates_pca,
        features=vel_rates_pca,
        k=15,
        stop_crit=0.0,
        delta=1.5,
        compute_laplacian=True,
        local_gauges=False,
    )

    return data, labels, pca


def find_sequences(vector):
    sequences = []
    start_index = 0

    for i in range(1, len(vector)):
        if vector[i] != vector[i - 1]:
            sequences.append((start_index, i - 1))
            start_index = i
    
    # Add the last sequence
    sequences.append((start_index, len(vector) - 1))

    return sequences

# Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.
def decoding_pos_dir(embedding_train, embedding_test, label_train, label_test):
    pos_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")
    dir_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")

    pos_decoder.fit(embedding_train, label_train[:,0])
    dir_decoder.fit(embedding_train, label_train[:,1])

    pos_pred = pos_decoder.predict(embedding_test)
    dir_pred = dir_decoder.predict(embedding_test)

    prediction = np.stack([pos_pred, dir_pred],axis = 1)

    test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)
    pos_test_err = np.median(abs(prediction[:,0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:,0])

    prediction_error = abs(prediction[:,0] - label_test[:, 0])

    # prediction error by back and forth
    sequences = find_sequences(label_test[:,1])
    errors = []
    for seq in sequences:
        errors.append(np.median(abs(prediction[seq,0] - label_test[seq, 0])))

    return test_score, pos_test_err, pos_test_score, prediction, prediction_error, np.array(errors)
