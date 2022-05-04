#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn.functional as F


def loss_comp(out):
    """
    Unsupervised loss function from Hamilton et al. 2018, using negative sampling.

    Parameters
    ----------
    out : pytorch tensor
        Output of network.
    Returns
    -------
    loss : float
        Loss.

    """
    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

    #loss function from word2vec
    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    loss = (-pos_loss - neg_loss)/out.shape[0]
    
    return loss