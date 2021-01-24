"""Kalman Filter"""
import numpy as np
from scipy.linalg import solve

class kalman():
    pass

def Kalman_n(x_0, H, B, W, G, C, V, y):
    """
    

    Parameters
    ----------
    x_0 : TYPE
        DESCRIPTION.
    H : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    W : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nT, dim = y.shape

    mu_pred = np.empty((nT,dim))
    sigma_pred = np.empty((nT, dim, dim))
    mu = np.empty((nT,dim))
    sigma = np.empty((nT, dim, dim))
    
    mu_pred[0,] = B+H@x_0
    sigma_pred[0,] = W@np.transpose(W)
    
    K=np.empty((nT, dim, dim))
    m = G@sigma_pred[0,]@np.transpose(G)+V@np.transpose(V)
    K[0,] = sigma_pred[0,]@np.transpose(G)@solve(m,np.identity(dim))
    
    mu[0,] = mu_pred[0,] + K[0,]@(y[0,]-C-G@mu_pred[0,])
    sigma[0,] = sigma_pred[0,]-K[0,]@G@sigma_pred[0,]
    
    for i in range(1,nT):
        mu_pred[i,] = B+H@mu[i-1,]
        sigma_pred[i,] = W@np.transpose(W)+H@sigma[i-1,]@np.transpose(H)
        m = G@sigma_pred[i,]@np.transpose(G)+V@np.transpose(V)
        K[i,] = sigma_pred[i,]@np.transpose(G)@solve(m,np.identity(dim))
    
        mu[i,] = mu_pred[i,] + K[i,]@(y[i,]-C-G@mu_pred[i,])
        sigma[i,] = sigma_pred[i,]-K[i,]@G@sigma_pred[i,]

    return(mu, sigma, K)

def Kalman_1(x_0, H, B, W, G, C, V, y):
    """
    

    Parameters
    ----------
    x_0 : TYPE
        DESCRIPTION.
    H : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    W : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nT = y.shape[0]

    mu_pred = np.empty(nT)
    sigma_pred = np.empty(nT)
    mu = np.empty(nT)
    sigma = np.empty(nT)
    
    mu_pred[0] = B+H*x_0
    sigma_pred[0] = W*W
    
    K=np.empty(nT)
    m = G*sigma_pred[0]*G+V*G
    K[0] = sigma_pred[0]*G/m
    
    mu[0] = mu_pred[0] + K[0]*(y[0]-C-G*mu_pred[0])
    sigma[0] = sigma_pred[0]-K[0]*G*sigma_pred[0]
    
    for i in range(1,nT):
        mu_pred[i] = B+H*mu[i-1]
        sigma_pred[i] = W*W+H*sigma[i-1]*H
        m = G*sigma_pred[i]*G+V*V
        K[i] = sigma_pred[i]*G/m
    
        mu[i] = mu_pred[i] + K[i]*(y[i]-C-G*mu_pred[i])
        sigma[i] = sigma_pred[i]-K[i]*G*sigma_pred[i]

    return(mu, sigma)
