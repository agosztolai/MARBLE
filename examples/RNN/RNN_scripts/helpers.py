import torch
import pickle
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
import seaborn as sns

from . import clustering
from . import modules
from . import dms


def load_network(f):
    """Load RNN network."""
    noise_std = 5e-2
    alpha = 0.2
    hidden_size = 500

    load = torch.load(f, map_location="cpu")

    if len(load) == 2:
        z, state = load
    else:
        state = load
        z = None

    net = modules.LowRankRNN(2, hidden_size, 1, noise_std, alpha, rank=2)
    net.load_state_dict(state)
    net.svd_reparametrization()

    return z, net


def sample_network(net, f):
    
    if os.path.exists(f):
        print('Network found with same name. Loading...')
        return torch.load(open(f, "rb"))

    n_pops = 2
    seed = 0
    z, _ = clustering.gmm_fit(net, n_pops, algo="bayes", random_state=seed)
    net_sampled = clustering.to_support_net(net, z)

    if os.path.exists(f):
        z, state = torch.load(f)
        net_sampled.load_state_dict(state)
    else:
        x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
        modules.train(
            net_sampled,
            x_train,
            y_train,
            mask_train,
            20,
            lr=1e-6,
            resample=True,
            keep_best=True,
            clip_gradient=1,
        )
        torch.save([z, net_sampled], f)

    return z, net_sampled


def generate_trajectories(
    net, input=None, epochs=None, n_traj=None, fname="./outputs/RNN_trajectories.pkl"
):

    if fname is not None:
        if os.path.exists(fname):
            return pickle.load(open(fname, "rb"))

    traj = []
    for i in range(len(input)):
        conds = []
        for _ in range(n_traj):
            # net.h0.data = torch.rand(size=net.h0.data.shape) #random ic
            _, traj_ = net(input[i].unsqueeze(0))
            traj_ = traj_.squeeze().detach().numpy()
            traj_epoch = [traj_[e : epochs[j + 1]] for j, e in enumerate(epochs[:-1])]
            conds.append(traj_epoch)

        traj.append(conds)

    pickle.dump(traj, open(fname, "wb"))

    return traj


def load_trajectories(fname):
    return pickle.load(open(fname, "rb"))


def plot_ellipse(ax, w, color="silver", std_factor=1):
    X = np.array([w[:, 0], w[:, 1]]).T
    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eig(cov)
    v1 = eigvecs[:, 0]
    angle = np.arctan(v1[1] / v1[0])
    angle = angle * 180 / np.pi
    ax.add_artist(
        Ellipse(
            xy=[0, 0],
            angle=angle,
            width=np.sqrt(eigvals[0]) * 2 * std_factor,
            height=np.sqrt(eigvals[1]) * 2 * std_factor,
            fill=True,
            fc=color,
            ec=color,
            lw=1,
            alpha=0.4,
        )
    )

    return ax


def plot_coefficients(net, z=None):
    if z is None:
        n_pops = 2
        z, _ = clustering.gmm_fit(net, n_pops, algo="bayes", random_state=0)

    m1 = net.m[:, 0].detach().numpy()
    n1 = net.n[:, 0].detach().numpy()
    m2 = net.m[:, 1].detach().numpy()
    n2 = net.n[:, 1].detach().numpy()
    wi1 = net.wi[0].detach().numpy()
    wi2 = net.wi[1].detach().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(12, 2))

    colors = ["#364285", "#E5BA52"]
    n_pops = 2
    clustering.pop_scatter_linreg(wi1, wi2, z, n_pops, colors=colors, ax=ax[0])
    plot_ellipse(
        ax[0], np.array([wi1[z.astype(bool)], wi2[z.astype(bool)]]).T, std_factor=3, color=colors[1]
    )
    plot_ellipse(
        ax[0],
        np.array([wi1[~z.astype(bool)], wi2[~z.astype(bool)]]).T,
        std_factor=3,
        color=colors[0],
    )

    clustering.pop_scatter_linreg(m1, m2, z, n_pops, colors=colors, ax=ax[1])
    clustering.pop_scatter_linreg(n1, n2, z, n_pops, colors=colors, ax=ax[2])
    clustering.pop_scatter_linreg(m1, n1, z, n_pops, colors=colors, ax=ax[3])


def setup_matplotlib():
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 19
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams["axes.titlepad"] = 24
    plt.rcParams["axes.labelpad"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 14


def get_lower_tri_heatmap(
    ov, bounds=None, figsize=None, cbar=False, cbar_shrink=0.9, cbar_pad=0.3, ax=None
):
    mask = np.zeros_like(ov, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ov[np.diag_indices_from(ov)] = 0
    # mask[np.diag_indices_from(mask)] = False
    mask = mask.T
    print(mask)
    if figsize is None:
        figsize = matplotlib.rcParams["figure.figsize"]

    if bounds is None:
        bound = np.max((np.abs(np.min(ov)), np.abs(np.max(ov))))
        bounds = [-bound, bound]

    # Set up the matplotlib figure
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, sep=10, as_cmap=True)
    print(ov)
    # Draw the heatmap with the mask and correct aspect ratio
    if not cbar:
        mesh = sns.heatmap(
            ov[:-1, 1:],
            mask=mask[:-1, 1:],
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar=False,
            vmin=bounds[0],
            vmax=bounds[1],
            ax=ax,
        )
    else:
        mesh = sns.heatmap(
            ov[:-1, 1:],
            mask=mask[:-1, 1:],
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar=True,
            vmin=bounds[0],
            vmax=bounds[1],
            ax=ax,
            cbar_kws={"shrink": cbar_shrink, "ticks": ticker.MaxNLocator(3), "pad": cbar_pad},
        )
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    return ax, mesh


def set_size(size, ax=None):
    """to force the size of the plot, not of the overall figure, from
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    w, h = size
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def center_limits(ax):
    xmin, xmax = ax.get_xlim()
    xbound = max(-xmin, xmax)
    ax.set_xlim(-xbound, xbound)
    ymin, ymax = ax.get_ylim()
    ybound = max(-ymin, ymax)
    ax.set_ylim(-ybound, ybound)


def plot_all_scatters(vectors):
    fig, ax = plt.subplots(len(vectors), len(vectors), figsize=(6, 8))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i is not j:
                ax[i, j].scatter(vectors[i], vectors[j])
    return ax


def plot_rates_single_neurons(rates, offset=1, colors=None, deltaT=1.0, figsize=(6, 8), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    cur_max = 0.0
    for i in range(rates.shape[1]):
        color = colors[i] if colors is not None else "red"
        ax.plot(
            np.arange(rates.shape[0]) * deltaT / 1000,
            rates[:, i] + cur_max + offset - np.min(rates[:, i]),
            color=color,
        )
        cur_max = cur_max + offset - np.min(rates[:, i]) + np.max(rates[:, i])
    return ax


def bar_plots_vectors(n, wi, wi_ctx1, wi_ctx2, title, xticks):
    fig, ax = plt.subplots()
    x = np.arange(3)
    ctx_derivative1 = phi_prime(wi_ctx1)
    ctx_derivative2 = phi_prime(wi_ctx2)
    win = wi
    Nfull = n.shape[0]
    neff_ctx1 = n.reshape(Nfull, 1) * ctx_derivative1.reshape(Nfull, 1)
    neff_ctx2 = n.reshape(Nfull, 1) * ctx_derivative2.reshape(Nfull, 1)
    ov1 = np.sum(n.reshape(Nfull, 1) * win.reshape(Nfull, 1))
    ov2 = np.sum(neff_ctx1.reshape(Nfull, 1) * win.reshape(Nfull, 1))
    ov3 = np.sum(neff_ctx2.reshape(Nfull, 1) * win.reshape(Nfull, 1))
    y = [ov1, ov2, ov3]
    ax.bar(x, y)
    plt.ylabel(title, fontsize=30)
    plt.xticks(x, xticks, fontsize=25)
    return ax


def radial_distribution_plot(x, N=80, bottom=0.1, cmap_scale=0.05, points=True):
    """
    Plot a radial histogram of angles
    :param x: if points=True, an array of shape nx2 of points in 2d space. if points=False a series of angles
    :param N: num bins
    :param bottom: radius of base circle
    :param cmap_scale: to adjust the colormap
    :param points: see x
    :return:
    """
    if points:
        assert len(x.shape) == 2 and x.shape[1] == 2
        x_cplx = np.array(x[:, 0], dtype=np.complex64)
        x_cplx.imag = x[:, 1]
        angles = np.angle(x_cplx)
    else:
        angles = x
    angles = angles % (2 * np.pi)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = [
        np.mean(np.logical_and(angles > theta[i], angles < theta[i + 1]))
        for i in range(len(theta) - 1)
    ]
    radii.append(np.mean(angles > theta[-1]))
    width = (2 * np.pi) / N
    offset = np.pi / N
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta + offset, radii, width=width, bottom=bottom)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / cmap_scale))
        bar.set_alpha(0.8)
    plt.yticks([])


def dimensionality_plot(trajectories, vecs, labels, figsize=None):
    """
    plot cumulative percentage of variance explained by vectors vecs, while ordering with most explicative vectors first
    :param trajectories: numpy array of shape #time_points x #neurons (with trials already flattened)
    :param vecs: list of numpy arrays of shape #neurons
    :param labels: labels associated with each vector
    :param figsize:
    :return: axes
    """
    total_var = np.sum(np.var(trajectories, axis=0))
    print(total_var)

    vars_nonorth = []
    for v in vecs:
        proj = trajectories @ v / np.linalg.norm(v)
        vars_nonorth.append(np.var(proj))
    indices = np.argsort(vars_nonorth)

    vecs_ordered = [vecs[i] for i in indices[::-1]]
    labels_ordered = [labels[i] for i in indices[::-1]]
    vecs_orth = gram_schmidt(vecs_ordered)
    variances = []
    for v in vecs_orth:
        proj = trajectories @ v / np.linalg.norm(v)
        variances.append(np.var(proj))
    print(variances)
    cumvar = np.cumsum(variances).tolist()
    print(cumvar)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        range(1, len(variances) + 1), cumvar / total_var * 100, color="lightslategray", alpha=0.5
    )
    ax.axhline(100, c="r")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(xticks=range(1, len(variances) + 1), xticklabels=labels_ordered, yticks=[0, 100])
    return ax


def phi_prime(x):
    return 1 - np.tanh(x) ** 2


def overlap_matrix(vectors):
    hidden_size = len(vectors[0])
    ov = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(i, len(vectors)):
            ov[i, j] = 1 / hidden_size * np.sum(vectors[i] * vectors[j])
    return ov


def boxplot_accuracies(accs, figsize=None, labels=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, acc in enumerate(accs):
        if not isinstance(acc, list):
            plt.scatter(i, acc, marker="*", s=90, c="k")
        else:
            bp = ax.boxplot(acc, positions=[i], widths=0.5)
            ax.scatter([i] * len(acc), acc, c="gray", alpha=0.5, s=5)
            [l.set_linewidth(2) for l in bp["medians"]]
    ax.set_xlim(-0.5, len(accs) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(list(range(len(accs))))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accuracy")
    ax.axhline(1, c="k", zorder=-10, lw=1, ls="--")
    return ax


def gram_schmidt(vecs):
    vecs_orth = []
    vecs_orth.append(vecs[0] / np.linalg.norm(vecs[0]))
    for i in range(1, len(vecs)):
        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]
        v = v / np.linalg.norm(v)
        vecs_orth.append(v)
    return vecs_orth


def gram_factorization(G):
    """
    The rows of the returned matrix are the basis vectors whose Gramian matrix is G
    :param G: ndarray representing a symmetric semidefinite positive matrix
    :return: ndarray
    """
    w, v = np.linalg.eigh(G)
    x = v * np.sqrt(w)
    return x


def angle(v, w, deg=True):
    res = np.arccos((v @ w) / (np.linalg.norm(v) * np.linalg.norm(w)))
    if not deg:
        return res
    else:
        return res * 180 / np.pi


def plot_experiment(net, input, traj, epochs, rect=(-8, 8, -6, 6), traj_to_show=1):
    fig, ax = plt.subplots(4, 5, figsize=(25, 20))
    idx = np.floor(np.linspace(0, len(input) - 1, 4))
    for i in range(4):
        for j, e in enumerate(epochs[:-1]):
            dms._plot_field(net, input[int(idx[i]), e], ax[i][j], rect=rect, sizes=1.3)
            epoch = [c[j] for c in traj[i]]
            dms.plot_trajectories(net, epoch, ax[i][j], c="#C30021", n_traj=traj_to_show)
            if j > 0:
                epoch = [c[j - 1] for c in traj[i]]
                dms.plot_trajectories(
                    net, epoch, ax[i][j], c="#C30021", style="--", n_traj=traj_to_show
                )

    ax[0][0].set_title("Fix")
    ax[0][1].set_title("Stim 1")
    ax[0][2].set_title("Delay")
    ax[0][3].set_title("Stim 2")
    ax[0][4].set_title("Decision")

    for i in range(4):
        ax[i][0].set_ylabel(r"$\kappa_2$")
    for i in range(5):
        ax[3][i].set_xlabel(r"$\kappa_1$")

    fig.subplots_adjust(hspace=0.1, wspace=0.1)


def aggregate_data(traj, epochs, transient=10, only_stim=False):

    n_conds = len(traj)
    n_epochs = len(epochs) - 1
    n_traj = len(traj[0])

    # fit PCA to all data
    pos = []
    for i in range(n_conds):  # conditions
        for k in range(n_epochs):
            for j in range(n_traj):  # trajectories
                pos.append(traj[i][j][k][transient:])

    pca = PCA(n_components=3)
    pca.fit(np.vstack(pos))
    print("Explained variance: ", pca.explained_variance_ratio_)

    # aggregate data under baseline condition (no input)
    pos, vel = [], []
    if not only_stim:
        for i in range(n_conds):  # conditions
            pos_, vel_ = [], []
            for k in [0, 2, 4]:
                for j in range(n_traj):  # trajectories
                    pos_proj = traj[i][j][k][transient:]
                    pos_proj = pca.transform(pos_proj)
                    pos_.append(pos_proj[:-1])  # stack trajectories
                    vel_.append(np.diff(pos_proj, axis=0))  # compute differences

            pos_, vel_ = np.vstack(pos_), np.vstack(vel_)  # stack trajectories
            pos.append(pos_)
            vel.append(vel_)

    # aggregate data under stimulated condition
    for i in range(n_conds):  # conditions
        pos_, vel_ = [], []
        for k in [1, 3]:
            for j in range(n_traj):  # trajectories
                pos_proj = traj[i][j][k][transient:]
                pos_proj = pca.transform(pos_proj)
                pos_.append(pos_proj[:-1])
                vel_.append(np.diff(pos_proj, axis=0))

        pos_, vel_ = np.vstack(pos_), np.vstack(vel_)  # stack trajectories
        pos.append(pos_)
        vel.append(vel_)

    return pos, vel
