import numpy as np
from scipy import stats
from math import sqrt
import torch
import multiprocessing as mp
from itertools import repeat
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from modules import SupportLowRankRNN
from helpers import center_axes


def phi_prime(x):
    return 1 - np.tanh(x)**2


def hard_thresh_linreg(vec1, vec2, inp, thresh=.5, label1='', label2=''):
    """
    plot a scatter of (vec1, vec2) points, separated in 2 populations according to the threshold inp > thresh,
    with linear regressions
    :param vec1: array of shape n
    :param vec2: array of shape n
    :param inp: array of shape n
    :param thresh: float
    :param label1: label for x axis
    :param label2: label for y axis
    :return:
    """
    idx1 = phi_prime(inp) < thresh
    idx2 = phi_prime(inp) > thresh

    plt.scatter(vec1[idx1], vec2[idx1], c='orange', label='saturated')
    plt.scatter(vec1[idx2], vec2[idx2], c='green', label='non saturated')

    xmin, xmax = vec1.min(), vec1.max()

    slope, intercept, r_value, p_value, std_err = stats.linregress(vec1[idx2],
                                                                   vec2[idx2])
    print("slope: %f    intercept: %f" % (slope, intercept))
    print("r-squared: %f" % (r_value ** 2))
    print("p-value: %f" % p_value)
    xs = np.linspace(xmin, xmax, 100)
    plt.plot(xs, slope * xs + intercept, color='green')

    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(vec1[idx1],
                                                                        vec2[idx1])
    print("slope: %f    intercept: %f" % (slope2, intercept2))
    print("r-squared: %f" % (r_value2 ** 2))
    print("p-value: %f" % p_value2)
    xs = np.linspace(xmin, xmax, 100)
    plt.plot(xs, slope2 * xs + intercept2, color='orange')
    plt.legend()

    slope, intercept, r_value, p_value, std_err = stats.linregress(vec1, vec2)
    print("slope: %f    intercept: %f" % (slope, intercept))
    print("r-squared: %f" % (r_value ** 2))
    print("p-value: %f" % p_value)
    xs = np.linspace(xmin, xmax, 100)
    plt.plot(xs, slope * xs + intercept, color='b')
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.show()

    return idx1, idx2


def gmm_fit(net, n_components, algo='bayes', n_init=50, random_state=None, mean_precision_prior=None,
            weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None):
    """
    fit a mixture of gaussians to a set of vectors
    :param net
    :param n_components: int
    :param algo: 'em' or 'bayes'
    :param n_init: number of random seeds for the inference algorithm
    :param random_state: random seed for the rng to eliminate randomness
    :return: vector of population labels (of shape n), best fitted model
    """
    neurons_fs = make_vecs(net)
    
    if isinstance(neurons_fs, list):
        X = np.vstack(neurons_fs).transpose()
    else:
        X = neurons_fs
    if algo == "em":
        model = GaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state)
    else:
        model = BayesianGaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state,
                                        init_params='random', mean_precision_prior=mean_precision_prior,
                                        weight_concentration_prior_type=weight_concentration_prior_type,
                                        weight_concentration_prior=weight_concentration_prior)
    model.fit(X)
    z = model.predict(X)
    return z, model


def make_vecs(net):
    """
    return a list of vectors (list of numpy arrays of shape n) composing a network
    """
    return [net.m[:, i].detach().cpu().numpy() for i in range(net.rank)] + \
           [net.n[:, i].detach().cpu().numpy() for i in range(net.rank)] + \
           [net.wi[i].detach().cpu().numpy() for i in range(net.input_size)] + \
           [net.wo[:, i].cpu().detach().numpy() for i in range(net.output_size)]


def gram_factorization(G):
    """
    The rows of the returned matrix are the basis vectors whose Gramian matrix is G
    :param G: ndarray representing a symmetric semidefinite positive matrix
    :return: ndarray
    """
    w, v = np.linalg.eigh(G)
    x = v * np.sqrt(w)
    return x


def to_support_net(net, z, new_size=None, take_means=False, scaling=False):
    X = np.vstack(make_vecs(net)).transpose()
    _, counts = np.unique(z, return_counts=True)
    n_components = counts.shape[0]
    weights = counts / net.hidden_size
    if take_means:
        means = np.vstack([X[z == i].mean(axis=0) for i in range(n_components)])
    else:
        means = np.zeros((n_components, X.shape[1]))
    covariances = [np.cov(X[z == i].transpose()) for i in range(n_components)]

    rank = net.rank
    basis_dim = 2 * rank + net.input_size + net.output_size
    m_init = torch.zeros(rank, n_components, basis_dim)
    n_init = torch.zeros(rank, n_components, basis_dim)
    wi_init = torch.zeros(net.input_size, n_components, basis_dim)
    wo_init = torch.zeros(net.output_size, n_components, basis_dim)

    # if new_size is None:
    #     new_size = net.hidden_size
    # if scaling:
    #     old_size = net.hidden_size
    # else:
    #     old_size = 1
    m_means = torch.from_numpy(means[:, :rank]).t() #* sqrt(old_size) / sqrt(new_size)
    n_means = torch.from_numpy(means[:, rank: 2*rank]).t() #* sqrt(old_size) / sqrt(new_size)
    wi_means = torch.from_numpy(means[:, 2*rank: 2*rank + net.input_size]).t()

    for i in range(n_components):
        # Compute Gramian matrix of the basis we have to build
        G = covariances[i]
        X_reduced = gram_factorization(G)
        for k in range(rank):
            m_init[k, i] = torch.from_numpy(X_reduced[k]) #* sqrt(old_size) / sqrt(new_size)
            n_init[k, i] = torch.from_numpy(X_reduced[rank + k]) #* sqrt(old_size) / sqrt(new_size)
        for k in range(net.input_size):
            wi_init[k, i] = torch.from_numpy(X_reduced[2 * rank + k])
        for k in range(net.output_size):
            wo_init[k, i] = torch.from_numpy(X_reduced[2 * rank + net.input_size + k]) #* old_size / new_size

    net2 = SupportLowRankRNN(net.input_size, new_size, net.output_size, net.noise_std, net.alpha, rank, n_components,
                             weights, basis_dim, m_init, n_init, wi_init, wo_init, m_means, n_means, wi_means)
    return net2


def to_support_net_old(net, n_components=1, new_size=None):
    vecs = make_vecs(net)
    z, model = gmm_fit(vecs, n_components)
    weights = model.weights_

    rank = net.rank
    basis_dim = 2 * rank + net.input_size + net.output_size
    m_init = torch.zeros(rank, n_components, basis_dim)
    n_init = torch.zeros(rank, n_components, basis_dim)
    wi_init = torch.zeros(net.input_size, n_components, basis_dim)
    wo_init = torch.zeros(net.output_size, n_components, basis_dim)

    if new_size is None:
        new_size = net.hidden_size
    old_size = net.hidden_size
    m_means = torch.from_numpy(model.means_[:, :rank]).t() * sqrt(old_size) / sqrt(new_size)
    n_means = torch.from_numpy(model.means_[:, rank: 2*rank]).t() * sqrt(old_size) / sqrt(new_size)
    wi_means = torch.from_numpy(model.means_[:, 2*rank: 2*rank + net.input_size]).t()

    for i in range(n_components):
        # Compute Gramian matrix of the basis we have to build
        G = model.covariances_[i]
        X_reduced = gram_factorization(G)
        for k in range(rank):
            m_init[k, i] = torch.from_numpy(X_reduced[k]) * sqrt(old_size) / sqrt(new_size)
            n_init[k, i] = torch.from_numpy(X_reduced[rank + k]) * sqrt(old_size) / sqrt(new_size)
        for k in range(net.input_size):
            wi_init[k, i] = torch.from_numpy(X_reduced[2 * rank + k])
        for k in range(net.output_size):
            wo_init[k, i] = torch.from_numpy(X_reduced[2 * rank + net.input_size + k]) * old_size / new_size

    net2 = SupportLowRankRNN(net.input_size, new_size, net.output_size, net.noise_std, net.alpha, rank, n_components,
                             weights, basis_dim, m_init, n_init, wi_init, wo_init, m_means, n_means, wi_means)
    return net2


def pop_scatter_linreg(vec1, vec2, pops, n_pops=None, linreg=True, colors=('blue', 'green', 'red', 'violet', 'gray'),
                       figsize=(5, 5), size=10., ax=None):
    """
    scatter plot of (vec1, vec2) points separated in populations according to int labels in vector pops, with linear
    regressions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    center_axes(ax)

    # Computing axes limits
    xmax = max(abs(vec1.min()), vec1.max())
    xmin = -xmax
    ax.set_xlim(xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin))
    ymax = max(abs(vec2.min()), vec2.min())
    ymin = -ymax
    ax.set_ylim(ymin - .1 * (ymax - ymin), ymax + .1 * (ymax - ymin))
    xs = np.linspace(xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin), 100)

    if n_pops is None:
        n_pops = np.unique(pops).shape[0]
    for i in range(n_pops):
        ax.scatter(vec1[pops == i], vec2[pops == i], color=colors[i], s=size)
        if linreg:
            slope, intercept, r_value, p_value, std_err = stats.linregress(vec1[pops == i], vec2[pops == i])
            print(f"pop {i}: slope={slope:.2f}, intercept={intercept:.2f}")
            ax.plot(xs, slope * xs + intercept, color=colors[i], zorder=-1)
    ax.set_xticks([])
    ax.set_yticks([])


def all_scatter_linreg(vecs, pops, xlabel='', ylabel='', n_pops=None, linreg=False,
                       colors=('blue', 'green', 'red', 'violet', 'gray')):
    fig, ax = plt.subplots(len(vecs),len(vecs),figsize=(6, 8))
    if n_pops is None:
        n_pops = np.unique(pops).shape[0]
    for k in range(len(vecs)):
        for l in range(len(vecs)):
            if k is not l:
                for i in range(n_pops):
                    ax[k,l].scatter(vecs[k][pops == i], vecs[l][pops == i], color=colors[i])
                    if linreg:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(vec1[pops == i], vec2[pops == i])
                        print(f"pop {i}: slope={slope:.2f}, intercept={intercept:.2f}")
                        plt.plot(xs, slope * xs + intercept, color=colors[i])
    
    return ax


### Spectral clustering and stability analysis

def generate_subsamples(neurons_fs, fraction=.8):
    indexes = np.random.choice(neurons_fs.shape[0], int(fraction * neurons_fs.shape[0]), replace=False)
    indexes = np.sort(indexes)
    return neurons_fs[indexes], indexes


def spectral_clustering(neurons_fs, n_clusters, metric='euclidean', n_neighbors=10):
    if metric == 'euclidean':
        model = SpectralClustering(n_clusters, affinity='nearest_neighbors')
        model.fit(neurons_fs)
    elif metric == 'cosine':
        model = SpectralClustering(n_clusters, affinity='precomputed')
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
        nn.fit(neurons_fs)
        knn_graph = nn.kneighbors_graph()
        knn_graph = 0.5 * (knn_graph + knn_graph.transpose())
        model.fit(knn_graph)
    return model


def clustering_stability_task(neurons_fs, algo, n_clusters, metric, n_neighbors, mean_precision_prior=1e5):
    sample, indexes = generate_subsamples(neurons_fs)
    if algo == 'spectral':
        model = spectral_clustering(sample, n_clusters, metric, n_neighbors)
        labels = model.labels_
    else:
        labels, _ = gmm_fit(sample, n_clusters, mean_precision_prior=mean_precision_prior,
                            weight_concentration_prior_type='dirichlet_distribution')
    return labels, indexes


def clustering_stability(neurons_fs, n_clusters, n_bootstrap, algo='gmm', metric='cosine', n_neighbors=10,
                         mean_precision_prior=1e5, normalize=None):
    """
    :param neurons_fs: numpy array of shape Nxd (neurons embedded in some feature space)
    :param n_clusters: int
    :param n_bootstrap: int
    :param algo: 'spectral' or 'gmm'
    :param metric: 'euclidean' or 'cosine' for spectral clustering
    :param n_neighbors: int, for spectral clustering
    :param mean_precision_prior:
    :param normalize: None, 'normal' or 'uniform'
    :return: list of n_bootstrap x (n_bootstrap - 1) / 2 ARI values for bootstrapped clusterings (possibly normalized)
    """

    with mp.Pool(mp.cpu_count()) as pool:
        args = repeat((neurons_fs, algo, n_clusters, metric, n_neighbors, mean_precision_prior), n_bootstrap)
        res = pool.starmap(clustering_stability_task, args)
        labels_list, indexings = zip(*res)

    aris = []
    # Align bootstrap samples and compute pairwise Rand indexes
    for i in range(n_bootstrap):
        for j in range(i+1, n_bootstrap):
            # build aligned labellings
            indexes_i = indexings[i]
            indexes_j = indexings[j]
            labels_i = []
            labels_j = []
            l = 0
            for k in range(len(indexes_i)):
                if l > len(indexes_j):
                    break
                while l < len(indexes_j) and indexes_j[l] < indexes_i[k]:
                    l += 1
                if l < len(indexes_j) and indexes_j[l] == indexes_i[k]:
                    labels_i.append(labels_list[i][k])
                    labels_j.append(labels_list[j][l])
                    l += 1
            aris.append(adjusted_rand_score(labels_i, labels_j))

    if normalize is not None:
        if normalize == 'normal':
            X_base = np.random.randn(neurons_fs.shape[0], neurons_fs.shape[1])
        elif normalize == 'uniform':
            X_base = (np.random.rand(neurons_fs.shape[0], neurons_fs.shape[1]) - 0.5) * 2
        base_aris = clustering_stability(X_base, n_clusters, n_bootstrap, metric, normalize=None)
        base_mean, base_std = np.mean(base_aris), np.std(base_aris)
        aris = [(ari - base_mean) / base_std for ari in aris]
    return aris


def boxplot_clustering_stability(neurons_fs, clusters_nums, aris=None, algo='gmm', n_bootstrap=20, metric='cosine',
                                 n_neighbors=10, ax=None):
    if aris is None:
        aris = [clustering_stability(neurons_fs, k, n_bootstrap, algo, metric, n_neighbors) for k in clusters_nums]
    aris = np.array(aris)
    if ax is None:
        fig, ax = plt.subplots()
    col_lines = 'indianred'
    bp = ax.boxplot(aris.T, patch_artist=True)
    for box in bp['boxes']:
        box.set(color='steelblue', facecolor='steelblue')
    for med in bp['medians']:
        med.set(color=col_lines)
    for l in bp['whiskers']:
        l.set(color=col_lines)
    for l in bp['caps']:
        l.set(color=col_lines)
    ax.set_xticks(list(range(1, aris.shape[0] + 1)))
    ax.set_xticklabels(clusters_nums)
    ax.set(xlabel='number of clusters', ylabel='stability', ylim=(-.1, 1.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax
