#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:35:46 2021

@author: wtredman
"""


def W2XY(W):
    X = W[:, :-1]
    Y = W[:, 1:]
    return X, Y

def ExactDMD(W): 
    X, Y = W2XY(W)
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)
    
    U, s, Vh = scipy.linalg.svd(X, full_matrices=False) 
    S = np.diag(s)
    U = np.asmatrix(U)
    S = np.asmatrix(S)
    Vh = np.asmatrix(Vh)

    A_tilde = U.H * Y * Vh.H * np.linalg.inv(S)

    lam, w = np.linalg.eig(A_tilde)
    order = np.argsort(lam)
    w = np.asmatrix(w)
    
    phi = (1./lam[order[-1]]) * Y * Vh.H * np.linalg.inv(S) * w[:, order[-1]]
    
    return phi