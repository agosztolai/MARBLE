"""This example illustrates MARBLE for a vector field on a flat surface."""
import numpy as np
import sys
from MARBLE import plotting, preprocessing, dynamics, net, postprocessing, geometry, distribution_distances
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def get_pos_vel(mus, 
                alpha=0.2, 
                n=250,
                t = np.array([0]),#np.arange(0, 3, 0.5),
                area = [[-3, -3],[3, 3]],
                radius=3,
                ):
    
    X0_range = dynamics.initial_conditions(n, len(mus), area=area, radius=radius, method='random', shape='rectangle')

    pos, vel = [], []
    for X0, m in zip(X0_range, mus):
        p, v = dynamics.simulate_vanderpol(m, X0, t)  
        pos.append(np.vstack(p))
        vel.append(np.vstack(v))

    pos, vel = dynamics.embed_parabola(pos, vel, alpha=alpha)
    rotation = []
    for i, (p, v) in enumerate(zip(pos,vel)):
        random_rotation = R.random(random_state=i)        
        p = random_rotation.apply(p)
        v = random_rotation.apply(v)  
        pos[i] = p
        vel[i] = v
        rotation.append(random_rotation.as_matrix())
        
    return pos, vel, rotation


def main():

    mus = np.linspace(-1,1,11)
    x, y, rot = get_pos_vel(mus)
    k = 10
    frac_geodesic_nb = 2
    
    # construct data object
    data = preprocessing.construct_dataset(x, y, k=k, frac_geodesic_nb=frac_geodesic_nb, local_gauges=False)
    
    params = {
        "lr":0.1,
        "order": 2,  # order of derivatives
        "include_self": True,#True, 
        "hidden_channels":[64],
        "out_channels": 3,
        "batch_size" : 64, # batch size
        #"emb_norm": True,
        "scalar_diffusion":False,
        "vector_diffusion":False,
        "include_positions":False, # don't / use positional features
        "epochs": 100,
        "inner_product_features": False,
        "global_align": True, # align dynamical systems orthogonally
        "final_grad": True, # compute orthogonal gradient at end of batch
        "positional_grad":True,  # use gradient on positions or not
        "vector_grad":True,
        "derivative_grad":True,
        "gauge_grad":True,
    }
    
    model = net(data, params=params)
    model.fit(data)

    # evaluate model on data
    data = model.transform(data)
    data = postprocessing.cluster(data)
    data = postprocessing.embed_in_2D(data)
    data = distribution_distances(data)
    data = postprocessing.rotate_systems(model, data)


    # plot aligned results
    plotting.fields(data, rotated=True,  col=2)
    plt.savefig('fields_rotated.png')

    # plot results
    plotting.fields(data,  col=2)
    plt.savefig('fields.png')
    plotting.embedding(data, mus[data.system.numpy().astype(int)])
    plt.savefig('embedding.png')
    
    plt.figure(figsize=(4, 4))
    im = plt.imshow(data.dist, extent=[mus[0], mus[-1], mus[0], mus[-1]])
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.colorbar(im, shrink=0.8)
    
    if params["out_channels"]>2:
        plotting.embedding_3d(data, data.system.numpy(), clusters_visible=True)
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
