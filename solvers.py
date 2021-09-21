#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.integrate import ode, odeint
import sys
import numpy as np
from ODE_library import *

def simulate_ODE(whichmodel, t, X0, P=None, noise=False, **noise_pars):
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
    P : dict, optional
        Parameters. The default is None.
    noise : bool, optional
        Add noise to solutions. Default is False.
    **noise_pars : additional keyword argument to specify noise parameters

    Returns
    -------
    x : len(t)xlen(X0) numpy array
        Trajectory.

    """
    
    f, jac = load_ODE(whichmodel, P=P)
    X = solve_ODE(f, jac, t, X0)
    
    if noise:
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


def generate_trajectories(whichmodel, n, t, X0_range, P=None, seed=None, noise=False, **noise_pars):
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
    P : dict, optional
        Parameters. The default is None.
    seed : int, optional
        Seed of random initial solutions. The default is None.
    noise : bool, optional
        Add noise to trajectories. Default is False.
    **noise_pars : additional keyword argument to specify noise parameters 

    Returns
    -------
    X : numpy array
        Trajectories.

    """
    
    if seed is not None:
        np.random.seed(seed)
        
    X = []
    for i in range(n):
        X0 = []
        for r in X0_range:
            X0.append(np.random.uniform(low=r[0], high=r[1]))
            
        X.append(simulate_ODE(whichmodel, t, X0, P=P, noise=noise, **noise_pars))
        
    return X


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
    ind = ind[int(t0*len(ind)):]
    ts = np.random.choice(ind, size=n, replace=True)
    
    X_sample = generate_flow(X, ts, T=T)
    
    t_sample = []
    for i in range(n):
        t_sample+=list(np.arange(0,T))
        
    return t_sample, X_sample


def generate_flow(X, ts, tt=None, T=10):
    """
    Obtain trajectories of between timepoints.

    Parameters
    ----------
    X : np array
        Trajectories.
    ts : int or np array or list[int]
        Source timepoint.
    tt : int or list[int]
        Target timepoint. The default is None.
    T : int
        Length of trajectory. Used when tt=None. The default is 10.

    Returns
    -------
    X_sample : list[np array].
        Set of flows of length T.

    """
    
    if tt is None:
        tt = [t+T for t in ts]
    else:
        assert len(tt)==len(ts), 'Number of source points must equal to the \
            number of target points.'
    
    X_sample = []
    for s,t in zip(ts,tt):
        X_sample.append(X[s:t+1,:])
        
    return X_sample


def load_ODE(whichmodel, P=None):
    """
    Load ODE system

    Parameters
    ----------
    whichmodel : sting
        ODE system from ODE_library.py.
    P : dict, optional
        Parameters. The default is None.

    Returns
    -------
    f : Callable
        ODE function.
    jac : Callable
        Jacobian.

    """

    if P == None:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)()
    else:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)(P)
                     
    return f, jac


def solve_ODE(f, jac, t, X0):
    
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

