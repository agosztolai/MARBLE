#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.integrate import ode, odeint
import sys
from ODE_library import *

def simulate_ODE(whichmodel, t, X0, P=None):
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

    Returns
    -------
    x : list
        Solution.
    xprime : list
        Time derivative of solution.

    """
    
    f, jac = load_ODE(whichmodel, P=P)
    X = solve_ODE(f, jac, t, X0)
    
    return X


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

