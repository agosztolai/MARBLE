#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Module to generate the spherical mnist data set'''

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from GeoDySys import geometry


def main():
    ''' '''
    
    chunk_size = 500
    bandwidth = 30
    noise = 1.0

    print("getting mnist data")
    trainset = datasets.MNIST(root='../data/MNIST', train=True, download=True)
    testset = datasets.MNIST(root='../data/MNIST', train=False, download=True)
    mnist_train = {}
    mnist_train['images'] = trainset.train_data.numpy()
    mnist_train['labels'] = trainset.train_labels.numpy()
    mnist_test = {}
    mnist_test['images'] = testset.test_data.numpy()
    mnist_test['labels'] = testset.test_labels.numpy()
    
    grid = geometry.grid_sphere(b=bandwidth)

    # result
    dataset = {}

    for label, data in zip(["train", "test"], [mnist_train, mnist_test]):

        print("projecting {0} data set".format(label))
        current = 0
        signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
        n_signals = signals.shape[0]
        proj_signals = np.ndarray(
            (signals.shape[0], 2 * bandwidth, 2 * bandwidth),
            dtype=np.uint8)

        while current < n_signals:

            # if not no_rotate[label]:
            #     rot = geometry.rand_rotation_matrix(deflection=noise)
            #     grid = geometry.rotate_grid(rot, grid)

            idxs = np.arange(current, min(n_signals,
                                          current + chunk_size))
            proj_signals[idxs] = geometry.project_2d_on_sphere(signals[idxs], grid)
            current += chunk_size
            print("\r{0}/{1}".format(current, n_signals), end="")
            
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # ax.view_init(elev=-50., azim=20)
            ax.plot_surface(grid[0], grid[1], grid[2], facecolors=plt.cm.coolwarm(proj_signals[0]), shade=False)
            break
        dataset[label] = {
            'images': proj_signals,
            'labels': data['labels']
        }
    # print("writing pickle")
    # with gzip.open("s2_mnist.gz", 'wb') as f:
    #     pickle.dump(dataset, f)

if __name__ == '__main__':
    main()