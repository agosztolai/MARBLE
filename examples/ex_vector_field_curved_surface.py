"""This example illustrates MARBLE for a vector field on a parabolic manifold."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt


def f0(x):
    return x * 0 + np.array([-1, -1])


def f1(x):
    return x * 0 + np.array([1, 1])


def f2(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] + 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] + 1) / norm
    return np.hstack([u, v])


def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] - 1) ** 2 + x[:, [1]] ** 2 + eps)
    u = x[:, [1]] / norm
    v = -(x[:, [0]] - 1) / norm
    return np.hstack([u, v])


def parabola(X, Y, alpha=0.05):
    Z = -((alpha * X) ** 2) - (alpha * Y) ** 2

    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


def main():

    # generate simple vector fields
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    n = 512
    x = [dynamics.sample_2d(n, [[-1, -1], [1, 1]], "random") for i in range(4)]
    y = [f0(x[0]), f1(x[1]), f2(x[2]), f3(x[3])]  # evaluated functions

    # embed on parabola
    for i, (p, v) in enumerate(zip(x, y)):
        end_point = p + v
        new_endpoint = parabola(end_point[:, 0], end_point[:, 1])
        x[i] = parabola(p[:, 0], p[:, 1])
        y[i] = (new_endpoint - x[i]) / np.linalg.norm(new_endpoint - x[i]) * np.linalg.norm(v)

    # construct PyG data object
    data = preprocessing.construct_dataset(
        x, y, graph_type="cknn", k=10, n_geodesic_nb=20, local_gauges=True  # use local gauges
    )

    # train model
    params = {
        "epochs": 70,  # optimisation epochs
        "order": 1,  # first-order derivatives are enough because the vector field have at most first-order features
        "hidden_channels": 32,  # 16 is enough in this simple example
        "out_channels": 3,  # 3 is enough in this simple example
        "inner_product_features": True,
    }

    model = net(data, params=params)
    model.run_training(data)

    # evaluate model on data
    data = model.evaluate(data)
    n_clusters = 20  # use 15 clusters for simple visualisation
    data = postprocessing.distribution_distances(data, n_clusters=n_clusters, cluster_typ="kmeans")
    data = postprocessing.embed_in_2D(data)

    # plot
    titles = ["Linear left", "Linear right", "Vortex right", "Vortex left"]
    # plot gauges in black to show that they 'hug' the manifold surface
    plotting.fields(data, titles=titles, col=2, width=3, scale=10, view=[0, 40], plot_gauges=True)
    # plt.savefig('../results/fields.svg')
    plotting.embedding(data, data.y.numpy(), titles=titles, clusters_visible=True)
    # plt.savefig('../results/embedding.svg')
    plotting.histograms(data, titles=titles)
    # plt.savefig('../results/histogram.svg')
    plotting.neighbourhoods(data)
    # plt.savefig('../results/neighbourhoods.svg')
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
