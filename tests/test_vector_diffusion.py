"""Test diffusion."""
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal

from MARBLE import construct_dataset
from MARBLE import dynamics
from MARBLE import geometry
from MARBLE import plotting
from MARBLE.layers import Diffusion


def f1(x):
    """f1"""
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] - 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] - 1) / norm
    return np.hstack([u, v])


def f2(x):
    """f2"""
    y = []
    for _ in range(x.shape[0]):
        y_ = np.random.uniform(size=(3))
        y_ /= np.linalg.norm(y_)
        y.append(y_)

    return np.vstack(y)


def sphere():
    """sphere"""
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 11j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T


def test_diffusion(plot=False):
    """Test diffusion and laplacian creation."""
    # parameters
    n = 512
    k = 30
    tau0 = 50

    # f1: constant, f2: linear, f3: parabola, f4: saddle
    x = dynamics.sample_2d(n, [[-1, -1], [1, 1]], "random")
    y = f1(x)  # evaluated functions

    # #construct PyG data object
    data = construct_dataset(x, y, graph_type="cknn", k=k)

    gauges, _ = geometry.compute_gauges(data)
    assert_array_almost_equal(
        gauges.numpy()[:5],
        np.array(
            [
                [[-0.19064367, 0.9816593], [0.9816593, 0.19064367]],
                [[-0.97356814, 0.22839674], [0.22839674, 0.97356814]],
                [[-0.91470975, -0.40411147], [-0.40411147, 0.91470975]],
                [[-0.1206701, 0.99269265], [0.99269265, 0.1206701]],
                [[-0.37872583, 0.9255089], [0.9255089, 0.37872583]],
            ]
        ),
        decimal=5,
    )
    R = geometry.compute_connections(data, gauges)
    assert_array_almost_equal(
        R.to_dense()[:5, :5],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, -0.22231616],
                [0.0, 1.0, 0.0, 0.0, -0.97497463],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [-0.22231616, -0.97497463, 0.0, 0.0, 1.0],
            ],
        ),
        decimal=5,
    )
    L = geometry.compute_laplacian(data)
    assert_array_almost_equal(
        L.to_dense().numpy()[:5, :5],
        np.array(
            [
                [1.0, 0.0, -0.01967779, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [-0.02420571, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ),
        decimal=5,
    )
    Lc = geometry.compute_connection_laplacian(data, R)
    assert_array_almost_equal(
        Lc.to_dense().numpy()[:5, :5],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.00437469],
                [0.0, 1.0, 0.0, 0.0, 0.01918535],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.00538132, 0.02359995, 0.0, 0.0, 1.0],
            ],
        ),
        decimal=5,
    )

    diffusion = Diffusion(tau0=tau0)
    data.x = diffusion(data.x, L, Lc, method="matrix_exp")
    assert_array_almost_equal(
        data.x.detach().numpy()[:5],
        np.array(
            [
                [0.4629867, 0.124888],
                [-0.03240441, 0.46744648],
                [-0.29019982, 0.40529823],
                [0.53165627, -0.00788487],
                [0.288114, 0.34210885],
            ]
        ),
        decimal=5,
    )

    if plot:
        plotting.fields(data)
        plt.show()


def test_diffusion_sphere(plot=False):
    """Test diffusion on sphere."""
    # parameters
    k = 0.4
    tau0 = 10.0

    x = sphere()
    y = f2(x)

    # construct PyG data object
    data = construct_dataset(x, y, graph_type="radius", k=k, n_geodesic_nb=10, var_explained=0.9)

    L = geometry.compute_laplacian(data)

    diffusion = Diffusion(tau0=tau0)
    data.x = diffusion(data.x, L, method="matrix_exp")
    assert_array_almost_equal(
        data.x.detach().numpy()[:5],
        np.array(
            [
                [0.513162, 0.44882008, 0.5685046],
                [0.35709542, 0.67346, 0.44372997],
                [0.32471117, 0.3551194, 0.81424344],
                [0.6844833, 0.53020036, 0.4575338],
                [0.5897326, 0.68115395, 0.41908088],
            ]
        ),
        decimal=5,
    )

    if plot:
        plotting.fields(data, alpha=1)
        plt.show()
