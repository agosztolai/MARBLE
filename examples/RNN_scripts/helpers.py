import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


def setup_matplotlib():
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 19
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['axes.titlepad'] = 24
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.size'] = 14


def get_lower_tri_heatmap(ov, bounds=None, figsize=None, cbar=False, cbar_shrink=.9, cbar_pad=.3, ax=None):
    mask = np.zeros_like(ov, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ov[np.diag_indices_from(ov)] = 0
    # mask[np.diag_indices_from(mask)] = False
    mask = mask.T
    print(mask)
    if figsize is None:
        figsize = matplotlib.rcParams['figure.figsize']

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
        mesh = sns.heatmap(ov[:-1, 1:], mask=mask[:-1, 1:], cmap=cmap, center=0, square=True, linewidths=.5,
                           cbar=False, vmin=bounds[0], vmax=bounds[1], ax=ax)
    else:
        mesh = sns.heatmap(ov[:-1, 1:], mask=mask[:-1, 1:], cmap=cmap, center=0, square=True, linewidths=.5,
                           cbar=True, vmin=bounds[0], vmax=bounds[1], ax=ax,
                           cbar_kws={"shrink": cbar_shrink, "ticks": ticker.MaxNLocator(3), 'pad': cbar_pad})
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    return ax, mesh


def set_size(size, ax=None):
    """ to force the size of the plot, not of the overall figure, from
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    w, h = size
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def center_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.set(xticks=[], yticks=[])


def remove_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set(xticks=[], yticks=[])


def center_limits(ax):
    xmin, xmax = ax.get_xlim()
    xbound = max(-xmin, xmax)
    ax.set_xlim(-xbound, xbound)
    ymin, ymax = ax.get_ylim()
    ybound = max(-ymin, ymax)
    ax.set_ylim(-ybound, ybound)



def plot_all_scatters(vectors):
    fig, ax = plt.subplots(len(vectors),len(vectors),figsize=(6, 8))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i is not j:
                ax[i,j].scatter(vectors[i],vectors[j])
    return ax


def plot_rates_single_neurons(rates, offset=1, colors=None, deltaT=1., figsize=(6, 8), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    cur_max = 0.
    for i in range(rates.shape[1]):
        color = colors[i] if colors is not None else 'red'
        ax.plot(np.arange(rates.shape[0]) * deltaT / 1000, rates[:, i] + cur_max + offset - np.min(rates[:, i]),
                color=color)
        cur_max = cur_max + offset - np.min(rates[:, i]) + np.max(rates[:, i])
    return ax


def bar_plots_vectors(n, wi, wi_ctx1, wi_ctx2, title, xticks):
    fig, ax = plt.subplots()
    x = np.arange(3)
    ctx_derivative1 = phi_prime(wi_ctx1)
    ctx_derivative2 = phi_prime(wi_ctx2)
    win = wi
    Nfull = n.shape[0]
    neff_ctx1 = n.reshape(Nfull,1) * ctx_derivative1.reshape(Nfull,1)
    neff_ctx2 = n.reshape(Nfull,1) * ctx_derivative2.reshape(Nfull,1)
    ov1 = np.sum(n.reshape(Nfull,1) * win.reshape(Nfull,1))
    ov2 = np.sum(neff_ctx1.reshape(Nfull,1) * win.reshape(Nfull,1))
    ov3 = np.sum(neff_ctx2.reshape(Nfull,1) * win.reshape(Nfull,1))
    y = [ov1, ov2, ov3]
    ax.bar(x, y)
    plt.ylabel(title, fontsize =30)
    plt.xticks(x, xticks,fontsize=25)
    return ax


def radial_distribution_plot(x, N=80, bottom=.1, cmap_scale=0.05, points=True):
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
    radii = [np.mean(np.logical_and(angles > theta[i], angles < theta[i+1])) for i in range(len(theta)-1)]
    radii.append(np.mean(angles > theta[-1]))
    width = (2*np.pi) / N
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
    ax.bar(range(1, len(variances) + 1), cumvar / total_var * 100, color='lightslategray', alpha=.5)
    ax.axhline(100, c='r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(xticks=range(1, len(variances) + 1), xticklabels=labels_ordered, yticks=[0, 100])
    return ax



def phi_prime(x):
    return 1 - np.tanh(x)**2


def map_device(tensors, net):
    """
    Maps a list of tensors to the device used by the network net
    :param tensors: list of tensors
    :param net: nn.Module
    :return: list of tensors
    """
    if net.wi.device != torch.device('cpu'):
        new_tensors = []
        for tensor in tensors:
            new_tensors.append(tensor.to(device=net.wi.device))
        return new_tensors
    else:
        return tensors


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
            plt.scatter(i, acc, marker='*', s=90, c='k')
        else:
            bp = ax.boxplot(acc, positions=[i], widths=.5)
            ax.scatter([i] * len(acc), acc, c='gray', alpha=.5, s=5)
            [l.set_linewidth(2) for l in bp['medians']]
    ax.set_xlim(-.5, len(accs)-.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(list(range(len(accs))))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])
    ax.set_ylabel('accuracy')
    ax.axhline(1, c='k', zorder=-10, lw=1, ls='--')
    return ax


def gram_schmidt_pt(mat):
    """
    Performs INPLACE Gram-Schmidt
    :param mat:
    :return:
    """
    mat[0] = mat[0] / torch.norm(mat[0])
    for i in range(1, mat.shape[0]):
        mat[i] = mat[i] - (mat[:i].t() @ mat[:i] @ mat[i])
        mat[i] = mat[i] / torch.norm(mat[i])


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

