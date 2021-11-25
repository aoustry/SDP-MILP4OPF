# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:29:17 2021

@author: aoust
"""

from scipy.optimize import minimize
import numpy as np
import time


def minsin(l,u):
    assert(l<u)
    res = minimize(np.sin, x0 = 0.5*(l+u), bounds = [(l,u)] , jac = np.cos, tol=1e-6)
    return np.sin(res.x)

def maxsin(l,u):
    assert(l<u)
    res = minimize(lambda x: -np.sin(x), x0 = 0.5*(l+u), bounds = [(l,u)] , jac = lambda x: -np.cos(x), tol=1e-6)
    return np.sin(res.x)

def mincos(l,u):
    assert(l<u)
    res = minimize(np.cos, x0 = 0.5*(l+u), bounds = [(l,u)] , jac = lambda x: -np.sin(x), tol=1e-6)
    return np.cos(res.x)

def minsin2(l,u):
    assert(l<u)
    assert(-2*np.pi<=l)
    assert(2*np.pi>=u)
    
    if (l<= -np.pi/2) and (-np.pi/2<=u):
        return -1
    
    if (l<= 3*np.pi/2) and (3*np.pi/2<=u):
        return -1
    
    return min(np.sin(l),np.sin(u))

def maxsin2(l,u):
    assert(l<u)
    
    assert(-2*np.pi<=l)
    assert(2*np.pi>=u)
    
    if (l<= np.pi/2) and (np.pi/2<=u):
        return 1
    if (l<= -3*np.pi/2) and (-3*np.pi/2<=u):
        return 1
    
    return max(np.sin(l),np.sin(u))

def mincos2(l,u):
    assert(-2*np.pi<=l)
    assert(2*np.pi>=u)
    if (l<= np.pi) and (np.pi<=u):
        return -1
    if (l<= -np.pi) and (-np.pi<=u):
        return -1
    return min(np.cos(l),np.cos(u))

def intersection(t1,t2):
    (l1,u1) = t1
    (l2,u2) = t2
    assert(l1<u1)
    assert(l2<u2)
    if u1<l2:
        return False
    if u2<l1:
        return False
    return True

# for i in range(100):
#     a,b = -2*np.random.rand(1)[0],2*np.random.rand(2)[0]
#     l,u = min(a,b), max(a,b)
#     t0 = time.time()
#     val1=minsin(l,u)
#     #print(time.time()-t0)
#     t0 = time.time()
#     val2=minsin2(l,u)
#     print(time.time()-t0)
#     assert(abs(val1-val2)<=1E-8)


    