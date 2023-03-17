import numpy as np
from matplotlib.colors import LightSource
from DE_library import simulate_ODE, simulate_trajectories
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import to_networkx
import networkx as nx
from example_utils import (
    initial_conditions,
    plot_phase_portrait,
    plot_phase_portrait,
    find_nn,
    circle,
)

from MARBLE import utils, geometry, net, plotting, postprocessing, compare_attractors


def parabola(X, Y, alpha=0.05):
    Z = -((alpha * X) ** 2) - (alpha * Y) ** 2
    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


def add_parabola(pos, vel):

    for i, (p, v) in enumerate(zip(pos, vel)):
        end_point = p + v
        new_endpoint = parabola(end_point[:, 0], end_point[:, 1])
        pos[i] = parabola(p[:, 0], p[:, 1])
        vel[i] = new_endpoint - pos[i]


def clean_data(pos, vel, _min=-3, _max=3):
    for i, (p, v) in enumerate(zip(pos, vel)):
        maskx = (p[:, 0] > _min) * (p[:, 0] < _max)
        masky = (p[:, 1] > _min) * (p[:, 1] < _max)
        mask = maskx & masky
        pos[i] = pos[i][mask]
        vel[i] = vel[i][mask]


if __name__ == "__main__":
    t1 = 1.0
    dt = 0.01
    t = np.arange(0, t1, dt)
    n = 50
    area = [[-2.5, -2.5], [2.5, 2.5]]

    mus = np.linspace(-0.5, 0.5, 11)
    positions = []
    velocities = []
    fig = plt.figure(figsize=(5, 4))
    for mu in mus:
        seed = np.random.randint(0, 10000)
        X0 = initial_conditions(n, 1, area, seed=seed)[0]
        pos, vel = simulate_trajectories("vanderpol", X0, t, par={"mu": mu})
        clean_data(pos, vel)
        add_parabola(pos, vel)

        positions.append(np.vstack(pos))
        velocities.append(np.vstack(vel))
        for _pos in pos:
            plt.scatter(_pos[0, 0], _pos[0, 1], c="k", s=2)
            plt.plot(_pos[:, 0], _pos[:, 1], c="k", lw=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-3, 3, -3, 3])
    plt.tight_layout()
    plt.savefig("multi_trajectories.pdf")

    data = utils.construct_dataset(positions, velocities, k=10, stop_crit=0.03, vector=False)

    plt.figure()

    G = to_networkx(
        data, node_attrs=["pos"], edge_attrs=None, to_undirected=True, remove_self_loops=True
    )
    pos = np.array([G.nodes[i]["pos"][:2] for i in G])
    signal = data.x.detach().numpy()
    nx.draw_networkx_nodes(G, pos=pos, node_color="k", node_size=4)
    nx.draw_networkx_edges(G, pos=pos, width=0.1)
    plt.quiver(pos[:, 0], pos[:, 1], signal[:, 0], signal[:, 1], color="k", width=0.004)

    plt.savefig("multi_parabola_graph.pdf")

    par = {
        "epochs": 75,  # optimisation epochs
        "order": 2,  # order of derivatives
        "hidden_channels": 32,  # number of internal dimensions in MLP
        "out_channels": 4,
        "inner_product_features": True,
    }

    model = net(data, par=par)
    model.run_training(data)

    data = model.evaluate(data)
    data = postprocessing(data, n_clusters=5)
    plotting.embedding(data)  # , labels=data.clusters['labels'], cbar_visible=False)
    plt.savefig("multi_embedding.pdf")
    plt.figure()
    plt.imshow(data.dist)
    plt.savefig("distance.pdf")
