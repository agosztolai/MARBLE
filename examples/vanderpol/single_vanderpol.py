import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import to_networkx
import networkx as nx

import MARBLE
from MARBLE import dynamics
from MARBLE import plotting


if __name__ == "__main__":
    t1 = 25
    dt = 0.05
    t = np.arange(0, t1, dt)
    n = 2
    area = [[-2.5, -2.5], [2.5, 2.5]]

    X0_range = dynamics.initial_conditions(n, 1, area, seed=11)[0]
    pos, vel = dynamics.simulate_vanderpol(0.5, X0_range, t)

    plotting.time_series(t, list(pos[0].T), style=".", figsize=(3, 2), ms=4)
    plt.savefig("time_series.pdf")
    plotting.time_series(t, list(pos[1].T), style=".", figsize=(3, 2), ms=4)
    plt.savefig("time_series_2.pdf")

    cycle = pos[0][-150:]

    fig = plt.figure(figsize=(5, 4))
    plt.plot(cycle[:, 0], cycle[:, 1], c="r", lw=1.5)

    plt.scatter(pos[0][0, 0], pos[0][0, 1], c="k")
    plt.plot(pos[0][:150, 0], pos[0][:150, 1], c="k", lw=0.8)
    plt.scatter(pos[1][0, 0], pos[1][0, 1], c="k")
    plt.plot(pos[1][:150, 0], pos[1][:150, 1], c="k", lw=0.8)

    pos, vel = dynamics.simulate_vanderpol(-0.5, X0_range, t)

    plt.scatter(pos[0][0, 0], pos[0][0, 1], c="b")
    plt.plot(pos[0][:150, 0], pos[0][:150, 1], c="b", lw=0.8)  # , ls="", marker=".")
    plt.scatter(pos[1][0, 0], pos[1][0, 1], c="b")
    plt.plot(pos[1][:150, 0], pos[1][:150, 1], c="b", lw=0.8)  # , ls="", marker=".")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-3, 3, -3, 3])
    plt.tight_layout()
    plt.savefig("illustration.pdf")

    t1 = 0.5
    dt = 0.05
    t = np.arange(0, t1, dt)
    n = 100

    X0_range = dynamics.initial_conditions(n, 1, area, seed=11)[0]
    pos, vel = dynamics.simulate_vanderpol(0.5, X0_range, t)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(cycle[:, 0], cycle[:, 1], c="r", lw=1.5)
    for _pos in pos:
        plt.scatter(_pos[0, 0], _pos[0, 1], c="k", s=2)
        plt.plot(_pos[:, 0], _pos[:, 1], c="k", lw=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([-3, 3, -3, 3])
    plt.tight_layout()
    plt.savefig("trajectories.pdf")

    for i, (p, v) in enumerate(zip(pos, vel)):
        end_point = p + v
        new_endpoint = dynamics.parabola(end_point[:, 0], end_point[:, 1])
        pos[i] = dynamics.parabola(p[:, 0], p[:, 1])
        vel[i] = new_endpoint - pos[i]

    data = MARBLE.construct_dataset(
        np.vstack(pos),
        features=np.vstack(vel),
        graph_type="cknn",
        k=10,
        stop_crit=0.03,
        vector=False,
    )

    plt.figure()

    G = to_networkx(
        data, node_attrs=["pos"], edge_attrs=None, to_undirected=True, remove_self_loops=True
    )
    pos = np.array([G.nodes[i]["pos"][:2] for i in G])
    plt.plot(cycle[:, 0], cycle[:, 1], c="r", lw=1.5)
    signal = data.x.detach().numpy()
    nx.draw_networkx_nodes(G, pos=pos, node_color="k", node_size=4)
    nx.draw_networkx_edges(G, pos=pos, width=0.1)
    plt.quiver(pos[:, 0], pos[:, 1], signal[:, 0], signal[:, 1], color="k", width=0.004)

    plt.savefig("parabola_graph.pdf")

    par = {
        "epochs": 75,  # optimisation epochs
        "order": 2,  # order of derivatives
        "hidden_channels": 32,  # number of internal dimensions in MLP
        "out_channels": 4,
        "inner_product_features": True,
    }

    model = MARBLE.net(data, par=par)
    model.run_training(data)

    data = model.evaluate(data)
    data = MARBLE.cluster_embeddings(data, n_clusters=5)
    plotting.embedding(data, labels=data.clusters["labels"], cbar_visible=False)
    plt.savefig("embedding.pdf")

    fig = plt.figure(figsize=(5, 4))
    plt.plot(cycle[:, 0], cycle[:, 1], c="r", lw=1.5)
    plt.scatter(data.pos[:, 0], data.pos[:, 1], c=data.clusters["labels"], s=5, cmap="tab20")
    plt.savefig("clusters.pdf")
