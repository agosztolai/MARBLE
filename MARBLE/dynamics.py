"""Dynamics module, adapted from DE_library.

TODO: clean this up
"""
import sys

import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint

from MARBLE import geometry


def fun_vanderpol(par=None):
    """Van der parol oscillator exhibiting a degenerate Hopf bifurcation"""
    if par is None:
        par = {"mu": 1.0}

    def f(_, X):
        x, y = X
        f1 = y
        f2 = par["mu"] * (1 - x**2) * y - x

        return [f1, f2]

    def jac(_, X):
        x, y = X
        df1 = [0.0, 1.0]
        df2 = [-2.0 * par["mu"] * x * y - 1.0, -par["mu"] * x**2]

        return [df1, df2]

    return f, jac


def load_ODE(whichmodel, par=None):
    """
    Load ODE system

    pararameters
    ----------
    whichmodel : sting
        ODE system from ODE_library.py.
    par : dict, optional
        pararameters. The default is None.

    Returns
    -------
    f : Callable
        ODE function.
    jac : Callable
        Jacobian.

    """

    if par is None:
        f, jac = getattr(sys.modules[__name__], f"fun_{whichmodel}")()
    else:
        f, jac = getattr(sys.modules[__name__], f"fun_{whichmodel}")(par)

    return f, jac


def solve_ODE(f, jac, t, x0, solver="standard"):
    """Solve ODE."""
    if solver == "standard":
        x = odeint(f, x0, t, Dfun=jac, tfirst=True)
        xprime = [f(t_, x_) for t_, x_ in zip(t, x)]

    elif solver == "zvode":
        r = ode(f, jac)
        r.set_integrator("zvode", method="bdf")
        # r.set_integrator('dopri5')
        r.set_initial_value(x0, t[0])

        # Run ODE integrator
        x = [x0]
        xprime = [f(0.0, x0)]

        for _t in t[1:]:
            r.integrate(_t)
            x.append(np.real(r.y))
            xprime.append(f(r.t, np.real(r.y)))

    return np.vstack(x), np.vstack(xprime)


def addnoise(X, **noise_pars):
    """
    Add noise to trajectories

    Parameters
    ----------
    X : np array
        Trajectories.
    **noise_pars : additional keyword argument to specify noise parameters

    Returns
    -------
    X : len(t)xlen(X0) numpy array
        Trajectory.

    """

    if noise_pars["noise"] == "Gaussian":
        mu = noise_pars["mu"]
        sigma = noise_pars["sigma"]
        X += np.random.normal(mu, sigma, size=X.shape)

    return X


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

    Returns
    -------
    X : list
        Solution.
    Xprime : list
        Time derivative of solution.

    """

    f, jac = load_ODE(whichmodel, par=par)
    X, Xprime = solve_ODE(f, jac, t, X0)

    if noise_pars:
        X = addnoise(X, **noise_pars)

    return X, Xprime


def simulate_trajectories(whichmodel, X0_range, t=1, par=None, **noise_pars):
    """
    Compute a number of trajectories from the given initial conditions.

    Parameters
    ----------
    Same as in simulate_ODE(), except:
    X0_range : list(list)
        List of initial conditions.

    Returns
    -------
    X_list : list(list)
        Solution for all trajectories.
    Xprime_list : list
        Time derivative of solution for all trajectories.

    """

    X_list, Xprime_list = [], []
    for X0 in X0_range:
        X, Xprime = simulate_ODE(whichmodel, t, X0, par=par, **noise_pars)
        X_list.append(X)
        Xprime_list.append(Xprime)

    return X_list, Xprime_list


def reject_outliers(*args, min_v=-3, max_v=3):
    """Reject outliers."""
    inds = []
    for arg in args:
        inds.append(np.where((arg > min_v).all(1) * (arg < max_v).all(1))[0])

    return list(set.intersection(*map(set, inds)))


def parabola(X, Y, alpha=0.05):
    """Parabola."""
    Z = -((alpha * X) ** 2) - (alpha * Y) ** 2

    return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])


def embed_parabola(pos, vel, alpha=0.05):
    """Embed on parabola."""
    for i, (p, v) in enumerate(zip(pos, vel)):
        end_point = p + v
        new_endpoint = parabola(end_point[:, 0], end_point[:, 1], alpha=alpha)
        pos[i] = parabola(p[:, 0], p[:, 1], alpha=alpha)
        vel[i] = new_endpoint - pos[i]
    return pos, vel


def sample_2d(N=100, interval=None, method="uniform", seed=0):
    """Sample N points in a 2D area."""
    if interval is None:
        interval = [[-1, -1], [1, 1]]
    if method == "uniform":
        x = np.linspace(interval[0][0], interval[1][0], int(np.sqrt(N)))
        y = np.linspace(interval[0][1], interval[1][1], int(np.sqrt(N)))
        x, y = np.meshgrid(x, y)
        x = np.vstack((x.flatten(), y.flatten())).T

    elif method == "random":
        np.random.seed(seed)
        x = np.random.uniform(
            (interval[0][0], interval[0][1]), (interval[1][0], interval[1][1]), (N, 2)
        )

    return x


def initial_conditions(n, reps, area=None, seed=0):
    """Generate iniital condition."""
    if area is None:
        area = [[-3, -3], [3, 3]]
    X0_range = [sample_2d(n, area, "random", seed=i + seed) for i in range(reps)]

    return X0_range


def simulate_vanderpol(mu, X0, t):
    """Simulate vanderpol."""
    p, v = simulate_trajectories("vanderpol", X0, t, par={"mu": mu})
    pos, vel = [], []
    for p_, v_ in zip(p, v):
        ind = reject_outliers(p_, v_)
        pos.append(p_[ind])
        vel.append(v_[ind])

    return pos, vel
