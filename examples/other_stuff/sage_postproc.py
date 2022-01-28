#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def model_vis(emb):
    colors = ["red", "orange", "green", "blue", "purple", "brown", "black"]
    colors += [colors[y] for y in data.y]
    xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
    plt.scatter(xs, ys, color=colors)