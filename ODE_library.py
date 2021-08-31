#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Library of standard dynamical systems

"""

def fun_saddle_node(P = {'mu': 1}):
    """Prototypical system exhibiting a saddle node bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = P['mu'] - x**2
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [-2*x, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    

def fun_trans_pitch(P = {'mu': 1}):
    """Prototypical system exhibiting a *transcritical* pitchfork bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = P['mu']*x - x**2
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [P['mu']-2*x, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    
    
def fun_sup_pitch(P = {'mu': 1}):
    """Prototypical system exhibiting a *supercritical* pitchfork bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = P['mu']*x - x**3
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [P['mu']-3*x**2, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac

    
def fun_sub_pitch(P = {'mu': 1}):
    """Prototypical system exhibiting a *subcritical* pitchfork bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = P['mu']*x + x**3
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [P['mu']+3*x**2, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    
    
def fun_hopf(P = {'beta': 1, 'sigma': -1}):
    """Prototypical system exhibiting a Hopf bifurcation at mu=0. 
    Supercritical for sigma=-1 and subcritical for sigma=+1"""
    
    def f(t, X):
        x, y = X
        f1 = P['beta']*x - y + P['sigma']*x*(x**2+y**2)
        f2 = x + P['beta']*y + P['sigma']*y*(x**2+y**2)
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [P['beta']+P['sigma']*(3*x**2+y**2), -1+2*P['sigma']*x*y]
        dfdy = [1+2*P['sigma']*y*x, P['beta']+P['sigma']*(3*y**2+x**2)]
        
        return [dfdx, dfdy]
    
    return f, jac


def fun_lotka_volterra(P = {'k': 0.3, 'c': 0.39}):
    """Lotka-Volterra model"""
    
    def f(t, X):
        x, y = X
        f1 = x*(1-x) - x*y/(x+P['c'])
        f2 = -P['k']*y + x*y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [1-2*x-y*(2*x+P['c'])/(x+P['c'])**2, -x/(x+P['c'])]
        dfdy = [y, -P['k'] + x]
        
        return [dfdx, dfdy]
    
    return f, jac
        

def fun_double_pendulum(P = {'b': 0.05, 'g': 9.81, 'l': 1.0, 'm': 1.0}):
    """Double pendulum"""
    
    def f(t, X):
        x, y = X
        f1 = y
        f2 = -(P['b']/P['m'])*y - P['g']*np.sin(x)
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [0.0, 1.0]
        dfdy = [-P['g']*np.cos(x), -P['b']/P['m']]
              
        return [dfdx, dfdy]
    
    return f, jac 


def fun_lorenz(P = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}):
    """Lorenz system"""
    
    def f(t, X):
        x, y, z = X
        f1 = P['sigma']*(y - x)/P['tau']
        f2 = (x*(P['rho'] - z) - y)/P['tau']
        f3 = (x*y - P['beta']*z)/P['tau']
        
        return [f1, f2, f3]
    
    def jac(t, X):
        x, y, z = X
        dfdx = [-P['sigma']/P['tau'], P['sigma']/P['tau'], 0.]
        dfdy = [(P['rho'] - z)/P['tau'], -1./P['tau'], -x/P['tau']]
        dfdz = [y/P['tau'], x/P['tau'], -P['beta']/P['tau']]
        
        return [dfdx, dfdy, dfdz]
                
    return f, jac            
    

def fun_rossler(P = {'a': 0.15, 'b': 0.2, 'c': 10.0}):
    """Rossler system"""
    
    def f(t, X):
        x, y, z = X
        f1 = -y - z
        f2 = x + P['a']*y
        f3 = P['b'] + z * (x - P['c'])
        
        return [f1, f2, f3]
    
    def jac(t, X):
        x, y, z = X
        dfdx = [0.,      -1, -1 ]
        dfdy = [1,   P['a'],  0.]
        dfdz = [z,       0.,  x ]
        
        return [dfdx, dfdy, dfdz]

    return f, jac


def fun_vanderpol(P = {'mu': 1.}):
    """Van der Pol oscillator, undergoes a degenerate Hofp bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = y
        f2 = P['mu']*(1-x**2)*y - x
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [0.,                1.    ]
        dfdy = [-2.*P['mu']*x*y - 1., -P['mu']*x**2]
        
        return [dfdx, dfdy]

    return f, jac


def fun_duffing(P = {'alpha', 'beta', 'gamma', 'delta', 'omega', 'tau'}):
    """Duffing oscillator"""
    
    def f(t, X):
        x, y, z = X
        f1 = y/P['tau']
        f2 = (-P['delta']*y - P['alpha']*x - P['beta']*x**3 + P['gamma']*np.cos(z))/P['tau']
        f3 = P['omega']
        
        return [f1, f2, f3]
    
    def jac(t, X):
        x, y, z = X
        dfdx = [0., 1./P['tau'], 0.]
        dfdy = [(-P['alpha'] - 3*P['beta']*x**2)/P['tau'], -P['delta']/P['tau'], -P['gamma']*np.sin(z)/P['tau']]
        dfdz = [0., 0., 0.]

        return [dfdx, dfdy, dfdz]

    return f, jac


def fun_kuramoto(P = {'W': np.array([28, 19, 11, 9, 2, 4]), 
                      'K': np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
                                     [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
                                     [-0.5, -0.5,  0,    1,   -0.5, -0.5],
                                     [-0.5, -0.5,  1,    0,   -0.5, -0.5],
                                     [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
                                     [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]])}    
                 ):
    """Kuramoto oscillator"""
    
#     def f(t, X):     
#         Xt = X[:, None]
#         dX = Xt - X
# #            if self.noise != None:
# #                n = self.noise().astype(self.dtype)
# #                phase += n
#         phase = P['W'] + np.sum(P['K']*np.sin(dX), axis=0)

#         return phase
    
    def f(t, X):
        Xt = X[:, None]
        dX = X-Xt
        phase = P['W'].astype(float)
        # if self.noise != None:
        #     n = self.noise().astype(self.dtype)
        #     phase += n
        phase += np.sum(P['K']*np.sin(dX),axis=1)

        return phase

    # def jac(t, X):
    #     Xt = X[:, None]
    #     dX = X - Xt
    #     phase = np.zeros(P['K'].shape)
    #     tmp = P['K']*np.cos(dX)
    #     tmp -= np.diag(tmp)
    #     phase += np.diag(np.sum(tmp, axis=0))
    #     phase -= tmp
        
    #     return phase
    
    def jac(t, X):

        Xt = X[:,None]
        dX = X-Xt
        
        # m_order = P['K'].shape[0]

        # phase = [m*P['K'][m-1]*np.cos(m*dX) for m in range(1,1+m_order)]
        phase = P['K']*np.cos(dX)
        phase = np.sum(phase, axis=0)

        for i in range(P['K'].shape[0]):
            phase[i,i] = -np.sum(phase[:,i])

        return phase

    return f, jac    


def fun_kuramoto_delay(P = {
                    'W': np.array([28, 19, 11, 9, 2, 4]), 
                    'K': np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
                                     [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
                                     [-0.5, -0.5,  0,    1,   -0.5, -0.5],
                                     [-0.5, -0.5,  1,    0,   -0.5, -0.5],
                                     [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
                                     [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]]),
                    'tau': 1}
                 ):
    """Kuramoto oscillator with delay"""
    
    def f(t, X):     
        Xt = X[:, None]
        dX = Xt - X
#            if self.noise != None:
#                n = self.noise().astype(self.dtype)
#                phase += n
        phase = P['W'] + np.sum(P['K']*np.sin(dX), axis=0)

        return phase

    return f
 
    
def fun_righetti_ijspeert(P = {'a', 'alpha', 'mu', 'K', 'omega_swing', 'omega_stance'}):
    """Righetti-Ijspeert phase oscillator"""

    def f(t, X):
        x = np.array(X[:6])
        y = np.array(X[6:])
            
        omega = P['omega_stance'] + (P['omega_swing'] - P['omega_stance']) / (1+np.exp(P['a']*y))            
        R = P['alpha']*(P['mu'] - x**2 - y**2)
            
        return (R*x - omega*y).tolist() + (R*y + omega*x + P['K'].dot(y)).tolist()

    def jac(t, X):
        x = np.array(X[:6])
        y = np.array(X[6:])
            
        omega = P['omega_stance'] + (P['omega_swing'] - P['omega_stance']) / (1+np.exp(P['a']*y))            
        R = P['alpha']*(P['mu'] - x**2 - y**2)
        
        dX = np.zeros([12,12])
        dX[6:11,6:11] = P['K']
        for i in range(6):
            dX[i,i]     = - 2*P['alpha']*x[i]**2 + R[i] 
            dX[i,i+6]   = - 2*P['alpha']*x[i]*y[i] - omega[i] + (omega[i]-P['omega_stance'])**2*P['a']*np.exp(P['a']*y[i])/(P['omega_swing']-P['omega_stance'])*y[i] 
            dX[i+6,i]   = - 2*P['alpha']*x[i]*y[i] + omega[i]
            dX[i+6,i+6] = - 2*P['alpha']*y[i]*2 + R[i] - (omega[i]-P['omega_stance'])**2*P['a']*np.exp(P['a']*y[i])/(P['omega_swing']-P['omega_stance'])*x[i]
            
        return dX.tolist()

    return f, jac