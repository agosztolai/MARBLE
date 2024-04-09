"""This example illustrates MARBLE for a vector field on a parabolic manifold."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def f0(x):
    return x * 0 + np.array([-1, -1])


def f1(x):
    return x * 0 + np.array([1, 1])


def f0(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] + 1) ** 2 + (x[:, [1]] + 1) ** 2 + eps)
    u = (x[:, [1]] + 1) / norm
    v = -(x[:, [0]] + 1) / norm
    return np.hstack([u, v])

def f1(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] - 1) ** 2 + (x[:, [1]] + 1) ** 2 + eps)
    u = (x[:, [1]] + 1) / norm
    v = -(x[:, [0]] - 1) / norm
    return np.hstack([u, v])

def f2(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] + 1) ** 2 + (x[:, [1]] -1) ** 2 + eps)
    u = (x[:, [1]]-1) / norm
    v = -(x[:, [0]] + 1) / norm
    return np.hstack([u, v])

def f3(x):
    eps = 1e-1
    norm = np.sqrt((x[:, [0]] - 1) ** 2 + (x[:, [1]] - 1) ** 2 + eps)
    u = (x[:, [1]]-1) / norm
    v = -(x[:, [0]] - 1) / norm
    return np.hstack([u, v])

# def f2(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]] + 1) ** 2 + x[:, [1]] ** 2 + eps)
#     u = x[:, [1]] / norm
#     v = -(x[:, [0]] + 1) / norm
#     return np.hstack([u, v])


# def f3(x):
#     eps = 1e-1
#     norm = np.sqrt((x[:, [0]] - 1) ** 2 + x[:, [1]] ** 2 + eps)
#     u = x[:, [1]] / norm
#     v = -(x[:, [0]] - 1) / norm
#     return np.hstack([u, v])


def parabola(X, Y, alpha=0.3):
    Z = -((alpha * X) ** 2) - (alpha * Y) ** 2

    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


def main():

    # generate simple vector fields
    # f0: linear, f1: point source, f2: point vortex, f3: saddle
    n = 512
    x = [dynamics.sample_2d(n, [[-1, -1], [1, 1]], "random") for i in range(3)]
    y = [f0(x[0]), 
         f1(x[1]), f2(x[2]),] #f3(x[3])]  # evaluated functions    

    alpha=0.05
    # embed on parabola
    for i, (p, v) in enumerate(zip(x, y)):
        end_point = p + v
        new_endpoint = parabola(end_point[:, 0], end_point[:, 1])
        x[i] = parabola(p[:, 0], p[:, 1], alpha=alpha)
        y[i] = (new_endpoint - x[i]) / np.linalg.norm(new_endpoint - x[i]) * np.linalg.norm(v)
        #alpha += 0.1

    rotate=True
    if rotate:
        for i, (p, v) in enumerate(zip(x,y)):
            random_rotation = R.random()        
            p = random_rotation.apply(p)
            v = random_rotation.apply(v)  
            x[i] = p
            y[i] = v

    # construct PyG data object
    data = preprocessing.construct_dataset(
        x, y, graph_type="cknn", k=15, local_gauges=False,  # use local gauges
    )

    # train model
    # params = {
    #     "order": 1,
    #     "inner_product_features": True,
    # }
    
    # train model
    params = {
        "lr":0.1,
        "order": 1,  # order of derivatives
        "out_channels": 2,
        "batch_size": 128,
        #"emb_norm": True,
        "include_positions":True,
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
    
    desired_layers = ['orthogonal']
    rotations_learnt = [param for i, (name, param) in enumerate(model.named_parameters()) if any(layer in name for layer in desired_layers)]
    
    pos_rotated = []
    vel_rotated = []
    for i, (p, v) in enumerate(zip(x,y)):
        rotation = rotations_learnt[i].cpu().detach().numpy() 
        p_rot = p @ rotation.T 
        v_rot = v @ rotation.T 
        #p_rot = rotation.dot(p.T).T
        #v_rot = rotation.dot(v.T).T
        pos_rotated.append(p_rot)
        vel_rotated.append(v_rot)

    data_ = preprocessing.construct_dataset(pos_rotated, vel_rotated, k=15, local_gauges=False)
    # plot results
    plotting.fields(data_,  col=2)
    plt.savefig('fields.png')

    # plot
    #titles = ["Linear left", "Linear right", "Vortex right", "Vortex left"]
    plotting.fields(data,  col=2)
    plt.savefig('fields.png')
    # plot gauges in black to show that they 'hug' the manifold surface
    #plotting.fields(data, titles=titles, col=2, width=3, scale=10, view=[0, 40], plot_gauges=True)
    #plt.savefig('fields.png')
    plotting.embedding(data, data.system.numpy(),  clusters_visible=True)
    plt.savefig('embedding.png')
    
    if params["out_channels"]>2:
        plotting.embedding_3d(data, data.system.numpy(),  clusters_visible=True)
        plt.savefig('embedding_3d.png')
        
    plotting.histograms(data,)
    plt.savefig('histogram.png')
    plotting.neighbourhoods(data)
    plt.savefig('neighbourhoods.png')
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
