import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import torch
from MARBLE import geometry
from MARBLE import utils
import matplotlib.pyplot as plt
from MARBLE.layers import AnisoConv


def f1(x, alpha):
    """Linear feature function"""
    return np.cos(alpha) * x[:, [0]] + np.sin(alpha) * x[:, [1]]


def f2(x, alpha):
    """Quadratic feature function"""
    return np.cos(alpha) * x[:, [0]] ** 2 - np.sin(alpha) * x[:, [1]] ** 2


def test_gauges(plot=False):
    """Test creation of local gauges."""
    n = 100
    k = 8
    alpha = np.pi / 4

    np.random.seed(1)
    x = np.random.uniform(low=(-1, -1), high=(1, 1), size=(n, 2))
    xv, yv = np.meshgrid(np.linspace(-1, 1, int(np.sqrt(n))), np.linspace(-1, 1, int(np.sqrt(n))))
    x = np.vstack([xv.flatten(), yv.flatten()]).T

    y = f1(x, alpha)
    y = torch.tensor(y)

    data = utils.construct_dataset(x, y, graph_type="cknn", k=k)
    gauges = data.gauges
    assert_array_equal(data.gauges, np.repeat(np.array([[[1.0, 0.0], [0.0, 1.0]]]), 100, axis=0))

    K = geometry.gradient_op(data.pos, data.edge_index, gauges)
    K = [utils.to_SparseTensor(_K.coalesce().indices(), value=_K.coalesce().values()) for _K in K]

    assert_array_almost_equal(
        K[0].to_scipy().toarray()[:5, :5],
        np.array(
            [
                [-1.0, 0.25, 0.5, 0.0, 0.0],
                [-0.16666667, -0.3333333, 0.16666667, 0.33333334, 0.0],
                [-0.33333334, -0.16666667, 0.3333333, 0.16666669, 0.0],
                [0.0, -0.25, -0.12500001, 0.0, 0.12500001],
                [0.0, 0.0, 0.0, -0.16666667, -0.3333333],
            ]
        ),
    )

    grad = AnisoConv()
    der = grad(y, K)
    assert_array_almost_equal(
        der.numpy()[:10],
        np.array(
            [
                [0.27498597, 0.27498597],
                [0.20951309, 0.15713481],
                [0.20951313, 0.15713481],
                [0.23570227, 0.15713482],
                [0.20951313, 0.15713482],
                [0.20951311, 0.15713483],
                [0.23570227, 0.15713483],
                [0.20951313, 0.15713484],
                [0.20951313, 0.15713484],
                [0.19641855, 0.19641855],
            ]
        ),
    )

    derder = grad(der, K)
    assert_array_almost_equal(
        derder.numpy()[:10],
        np.array(
            [
                [-7.85674201e-02, -1.17851151e-01, -1.17851155e-01, -7.85674242e-02],
                [-2.18240134e-03, -5.23782543e-02, -2.83715625e-02, 1.74594472e-02],
                [-1.74594164e-02, -5.23782863e-02, -3.92837183e-02, 2.34149155e-08],
                [6.43910189e-09, -7.85674248e-02, 6.43910197e-09, 2.10734241e-08],
                [4.36484562e-03, -5.23782887e-02, 7.80497299e-10, 1.87319325e-08],
                [-4.36486123e-03, -5.23782699e-02, 5.46348028e-09, 1.63904412e-08],
                [1.75611907e-09, -7.85674201e-02, 6.43910208e-09, 1.40489496e-08],
                [-8.72971660e-03, -5.23782770e-02, 1.30945743e-02, 1.17074580e-08],
                [-1.09121464e-02, -5.23782762e-02, 1.52770041e-02, 1.74594379e-02],
                [-7.02447468e-09, -3.92837112e-02, 3.92837112e-02, 8.19522059e-09],
            ]
        ),
    )

    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharey=True, figsize=(14, 3), subplot_kw={"aspect": 1}
        )
        ax1.scatter(x[:, 0], x[:, 1], c=y)
        ax1.set_title(r"$(f_x,f_y)$")
        ax1.axis("off")
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        ax2.scatter(x[:, 0], x[:, 1], c=y)
        ax2.set_title(r"$f_{xx}$,$f_{yy}$")
        ax2.axis("off")
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax3.scatter(x[:, 0], x[:, 1], c=y)
        ax3.set_title(r"$f_{xy}$,$f_{yx}$")
        ax3.axis("off")
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        for ind in range(x.shape[0]):
            ax1.arrow(x[ind, 0], x[ind, 1], der[ind, 0], der[ind, 1], width=0.01)
            ax2.arrow(x[ind, 0], x[ind, 1], derder[ind, 0], 0, width=0.01, color="r")
            ax2.arrow(x[ind, 0], x[ind, 1], 0, derder[ind, 3], width=0.01, color="b")
            ax3.arrow(x[ind, 0], x[ind, 1], derder[ind, 1], 0, width=0.01, color="r")
            ax3.arrow(x[ind, 0], x[ind, 1], 0, derder[ind, 2], width=0.01, color="b")

        PCM = ax1.get_children()[0]  # get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax1)
    y = f2(x, alpha)
    y = torch.tensor(y)

    der = grad(y, K)
    assert_array_almost_equal(
        der.numpy()[:5, :5],
        np.array(
            [
                [-3.14269681e-01, 3.14269681e-01],
                [-2.79350844e-01, 3.02630063e-01],
                [-2.79350835e-01, 3.02630057e-01],
                [-1.57134847e-01, 3.02630053e-01],
                [1.45692810e-08, 3.02630051e-01],
            ]
        ),
    )
    derder = grad(der, K)
    assert_array_almost_equal(
        derder.numpy()[:5, :5],
        np.array(
            [
                [4.36485566e-02, 2.61891408e-02, -2.61891408e-02, -4.36485566e-02],
                [6.78977669e-02, 3.87987339e-02, -7.75974289e-03, -4.65584549e-02],
                [5.52881903e-02, 1.04756561e-01, -3.87987803e-03, -5.81980716e-02],
                [1.22216001e-01, 5.23782832e-02, -4.29273473e-09, -5.81980696e-02],
                [1.01846653e-01, -3.49188700e-02, -7.97841570e-09, -5.81980685e-02],
            ]
        ),
    )

    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharey=True, figsize=(14, 3), subplot_kw={"aspect": 1}
        )
        ax1.scatter(x[:, 0], x[:, 1], c=y)
        ax1.set_title(r"$(f_x,f_y)$")
        ax1.axis("off")
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        ax2.scatter(x[:, 0], x[:, 1], c=y)
        ax2.set_title(r"$f_{xx}$,$f_{yy}$")
        ax2.axis("off")
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax3.scatter(x[:, 0], x[:, 1], c=y)
        ax3.set_title(r"$f_{xy}$,$f_{yx}$")
        ax3.axis("off")
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        for ind in range(x.shape[0]):
            ax1.arrow(x[ind, 0], x[ind, 1], der[ind, 0], der[ind, 1], width=0.01)
            ax2.arrow(x[ind, 0], x[ind, 1], derder[ind, 0], 0, width=0.01, color="r")
            ax2.arrow(x[ind, 0], x[ind, 1], 0, derder[ind, 3], width=0.01, color="b")
            ax3.arrow(x[ind, 0], x[ind, 1], derder[ind, 1], 0, width=0.01, color="r")
            ax3.arrow(x[ind, 0], x[ind, 1], 0, derder[ind, 2], width=0.01, color="b")

        PCM = ax1.get_children()[0]  # get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax1)
        plt.show()
