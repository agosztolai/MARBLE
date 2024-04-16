"""This example illustrates MARBLE for a vector field on a flat surface."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt
import scipy as sc

def f0(x):
    return x * 0 + np.array([-1, -1])

def f1(x):
    return x * 0 + np.array([1, 1])

def f2(x):
    return x * 0 + np.array([-1, 0])

def f3(x):
    return x * 0 + np.array([1, 0])


# def f2(x):
#     return x * 0 + np.array([1, -1])

# def f3(x):
#     return x * 0 + np.array([-1, 1])


# def f0(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]] + 1) ** 2 + (x[:, [1]] + 1) ** 2 + eps)
#     u = (x[:, [1]] + 1) / norm
#     v = -(x[:, [0]] + 1) / norm
#     return np.hstack([u, v])

# def f1(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]] - 1) ** 2 + (x[:, [1]] + 1) ** 2 + eps)
#     u = (x[:, [1]] + 1) / norm
#     v = -(x[:, [0]] - 1) / norm
#     return np.hstack([u, v])

# def f2(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]] + 1) ** 2 + (x[:, [1]] - 1) ** 2 + eps)
#     u = (x[:, [1]]-1) / norm
#     v = -(x[:, [0]] + 1) / norm
#     return np.hstack([u, v])

# def f3(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]] - 1) ** 2 + (x[:, [1]] - 1) ** 2 + eps)
#     u = (x[:, [1]]-1) / norm
#     v = -(x[:, [0]] - 1) / norm
#     return np.hstack([u, v])


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


def main():

    # generate simple vector fields
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    n = 256
    x = [dynamics.sample_2d(n, [[-1, -1], [1, 1]], "random", seed=i) for i in range(4)]
    y = [f0(x[0]), f1(x[1]),  f2(x[2]), f3(x[3])]  # evaluated functions

    # construct data object
    data = preprocessing.construct_dataset(x, y, local_gauges=False)

    # train model
    params = {
        "lr":0.1,
        "order": 2,  # order of derivatives
        "include_self": True,#True, 
        "hidden_channels":[64],
        "out_channels": 2,
        "batch_size": 64, # batch size
        #"emb_norm": True,
        "scalar_diffusion":False, # diffusion with graph Laplacian
        "vector_diffusion": False, # diffusion over connection Laplacian
        "include_positions":False, # don't / use positional features
        "epochs": 50,
        "inner_product_features":False, # compute inner product of features
        "global_align":True, # align dynamical systems orthogonally
        "final_grad": True, # compute orthogonal gradient at end of batch
        "positional_grad":False,  # compute orthogonal gradient on positions or not
        "vector_grad":True, # compute gradient based on cosine difference of systems
        "derivative_grad":True, 
        "gauge_grad":False, # use the normal vectors of the local gauges for gradient
    }
    
    model = net(data, params=params)
    model.fit(data)

    # evaluate model on data
    data = model.transform(data)
    data = postprocessing.cluster(data)
    data = postprocessing.embed_in_2D(data)
    data = postprocessing.rotate_systems(model, data)

    # plot aligned results
    plotting.fields(data, rotated=True,  col=2)
    plt.savefig('fields.png')

    # plot results
    titles = ["Linear left", "Linear right", "Vortex right", "Vortex left"]
    plotting.fields(data, col=2)
    plt.savefig('fields.png')
    plotting.embedding(data, data.system.numpy(), clusters_visible=True)
    plt.savefig('embedding.png')
    
    if params["out_channels"]>2:
        plotting.embedding_3d(data, data.system.numpy(), clusters_visible=True)
        plt.savefig('embedding_3d.png')
        
    plotting.histograms(data,)
    plt.savefig('histogram.png')
    plotting.neighbourhoods(data)
    plt.savefig('neighbourhoods.png')
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
