#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.integrate import ode, odeint
import sys
import numpy as np
from GeoDySys.ODE_library import *

def simulate_ODE(whichmodel, t, X0, par=None, **noise_pars):
    """
    Load ODE functions and run appropriate solver

    Parameters
    ----------
    whichmodel : string
        ODE system from ODE_library.py.
    t : array or list
        Time steps to evaluate system at.
    x0 : array or list
        Initial condition. Size must match the dimension of the ODE system.
    par : dict, optional
        Parameters. The default is None.
    noise : bool, optional
        Add noise to solutions. Default is False.
    **noise_pars : additional keyword argument to specify noise parameters

    Returns
    -------
    x : len(t)xlen(X0) numpy array
        Trajectory.

    """
    
    f, jac = load_ODE(whichmodel, par=par)
    X = solve_ODE(f, jac, t, X0)
    
    if noise_pars!={}:
        X = addnoise(X, **noise_pars)
    
    return X


def addnoise(X, noise_type='Gaussian', **noise_pars):
    """
    Add noise to trajectories

    Parameters
    ----------
    X : np array
        Trajectories.
    noise_type : string, optional
        Type of noise. The default is 'Gaussian'.
    **noise_pars : additional keyword argument to specify noise parameters

    Returns
    -------
    X : len(t)xlen(X0) numpy array
        Trajectory.

    """
    
    if noise_type=='Gaussian':
        mu = noise_pars['mu']
        sigma = noise_pars['sigma']
        X += np.random.normal(mu, sigma, size = X.shape)
        
    return X


def generate_trajectories(whichmodel, n, t, X0_range, par=None, stack=True, transient=None, seed=None, **noise_pars):
    """
    Generate an ensemble of trajectories from different initial conditions, 
    chosen randomly from a box.

    Parameters
    ----------
    whichmodel : string
        ODE system from ODE_library.py.
    n : int
        Number of trajectories to generate.
    t : array or list
        Time steps to evaluate system at..
    X0_range : list[list] e.g. [[0,1],[0,1],[0,1]]
        Lower/upper limits in each dimension of the box of initial solutions.
    par : dict, optional
        Parameters. The default is None.
    stack : bool, optional
        Stack trajectories into a long array. The default is True.
    transient : float between 0 and 1
        Ignore first transient fraction of time series.
    seed : int, optional
        Seed of random initial solutions. The default is None.
    noise : bool, optional
        Add noise to trajectories. Default is False.
    **noise_pars : additional keyword argument to specify noise parameters 

    Returns
    -------
    X_ens : numpy array
        Trajectories.
    t_ens : numpy array
        Time indices

    """
    
    if seed is not None:
        np.random.seed(seed)
        
    t_ens, X_ens = [], []
    for i in range(n):
        X0 = []
        for r in X0_range:
            X0.append(np.random.uniform(low=r[0], high=r[1]))
            
        X_ens.append(simulate_ODE(whichmodel, t, X0, par=par, **noise_pars))
        t_ens.append(np.arange(len(t)))
        
        if transient is not None:
            l_tr = int(len(X_ens[-1])*transient)
            X_ens[-1] = X_ens[-1][l_tr:]
            t_ens[-1] = t_ens[-1][:-l_tr]
        
    if stack:
        X_ens = np.vstack(X_ens)
        t_ens = np.hstack(t_ens)
        
    return t_ens, X_ens


def sample_trajectories(X, n, T, t0=0.1, seed=None):
    """
    Randomly sample trajectories from the attractor.

    Parameters
    ----------
    X : numpy array
        Trajectory including transient.
    n : int
        Number of trajectories to sample.
    T : int
        Length of trajectories (timesteps).
    t0 : float
        Initial transient fraction of the time series. The default is 0.1 (10%).
    seed : int, optional
        Seed of random initial solutions. The default is None.

    Returns
    -------
    t_sample : list(list)
        Time indices in the original attrator
    X_sample : list[array]
        n sampled trajectories.

    """
    
    if seed is not None:
        np.random.seed(seed)
    
    #Discard transient
    ind = np.arange(X.shape[0])
    ind = ind[int(t0*len(ind)):len(ind)-T]
    ts = np.random.choice(ind, size=n, replace=True)
    
    X_sample = generate_flow(X, ts, T=T)
    
    t_sample = []
    for i in range(n):
        t_sample+=list(np.arange(0,T))
        
    return t_sample, X_sample


def load_ODE(whichmodel, par=None):
    """
    Load ODE system

    Parameters
    ----------
    whichmodel : sting
        ODE system from ODE_library.py.
    par : dict, optional
        Parameters. The default is None.

    Returns
    -------
    f : Callable
        ODE function.
    jac : Callable
        Jacobian.

    """

    if par is None:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)()
    else:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)(par)
                     
    return f, jac


def solve_ODE(f, jac, t, X0):
    """
    ODE solver. May wanna change it to solve IVP

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    jac : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    X0 : TYPE
        DESCRIPTION.

    Returns
    -------

    """
    
    X = odeint(f, X0, t, Dfun=jac, tfirst=True)
    
#     r = ode(f, jac)
# #    r.set_integrator('zvode', method='bdf')
#     r.set_integrator('dopri5')
#     r.set_initial_value(x0, t[0])
      
#     #Run ODE integrator
#     x = [x0]
#     xprime = [f(0.0, x0)]
    
#     for idx, _t in enumerate(t[1:]):
#         r.integrate(_t)
#         x.append(np.real(r.y))
#         xprime.append(f(r.t, np.real(r.y)))    

    return X

