"""This example illustrates MARBLE for a vector field on a flat surface."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing, geometry, distribution_distances
import matplotlib.pyplot as plt


def get_pos_vel(mus, 
                alpha=0.05, 
                n=100,
                t = np.arange(0, 3, 0.5),
                area = [[-3, -3],[3, 3]]
                ):
    X0_range = dynamics.initial_conditions(n, len(mus), area)

    pos, vel = [], []
    for X0, m in zip(X0_range, mus):
        p, v = dynamics.simulate_vanderpol(m, X0, t)
        pos.append(np.vstack(p))
        vel.append(np.vstack(v))

    pos, vel = dynamics.embed_parabola(pos, vel, alpha=alpha)
    return pos, vel


def main():


    mus = np.linspace(-1,1,21)
    x, y = get_pos_vel(mus)

    # construct data object
    data = preprocessing.construct_dataset(x, y)

    # train model
    params = {
        "lr":0.1,
        "order": 2,  # order of derivatives
        "include_self": True,#True, 
        "hidden_channels":[64,64],
        "out_channels": 2,
        "batch_size" : 64, # batch size
        #"emb_norm": True,
        #"include_positions":True,
        "epochs":100,
        "inner_product_features":False,
        "global_align":True,
    }

    model = net(data, params=params)
    model.fit(data)

    # evaluate model on data
    data = model.transform(data)
    data = postprocessing.cluster(data)
    data = postprocessing.embed_in_2D(data)
    data = distribution_distances(data)


    # plot results
    plotting.fields(data,  col=2)
    plt.savefig('fields.png')
    plotting.embedding(data, mus[data.y.numpy().astype(int)])
    plt.savefig('embedding.png')
    
    plt.figure(figsize=(4, 4))
    im = plt.imshow(data.dist, extent=[mus[0], mus[-1], mus[0], mus[-1]])
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.colorbar(im, shrink=0.8)
    
    if params["out_channels"]>2:
        plotting.embedding_3d(data, data.y.numpy(), clusters_visible=True)
        plt.savefig('embedding_3d.png')
        
    emb_MDS, _ = geometry.embed(data.dist, embed_typ='MDS')
    plt.figure(figsize=(4, 4))
    plotting.embedding(emb_MDS, mus, s=30, alpha=1)
        
    plotting.histograms(data,)
    plt.savefig('histogram.png')
    plotting.neighbourhoods(data)
    plt.savefig('neighbourhoods.png')
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
