import numpy as np
from scipy.stats import poisson
from scipy.stats import multivariate_normal
import re
from scipy.stats import gaussian_kde

from particles import particles

def log_psi_tilde_t(x_prev, a, b, s, m_psi, s_psi):
    """
    Function for computing the log of the scaling factor \Tilde{\psi}(x_t)

    Parameters
    ----------
    x_prev : np.array or float
        x_t.
    a : np.array
        Constant term in the state transition.
    b : np.array
        Scaling term in state transition.
    s : np.array
        covariance matrix in state transition.
    m_psi : np.array
        Mean of gaussian \psi.
    s_psi : np.array
        Covariance matrix of gaussian \psi.

    Returns
    -------
    log_I : np.array or float
        The log of  the scaling factor.

    """
    m = a + b@x_prev
    s_twisted = np.linalg.inv(np.linalg.inv(s)+ np.linalg.inv(s_psi))
    m_twisted = s_twisted@(np.linalg.inv(s)@np.transpose(m) + np.linalg.inv(s_psi)@np.transpose(m_psi))
    
    #0.5*np.log(2*np.pi*np.linalg.det(s_twisted))
    log_I = -0.5*np.log(np.linalg.det(2*np.pi*(s+s_psi)))-0.5*(m-m_psi)@np.linalg.inv(s+s_psi)@np.transpose(m-m_psi)
   
    return log_I


def psi_pf_n(n, psi_mean, psi_cov, x_0, y, emission_dens, state_sample, N = 1000, seed = 123, res_rate = 0.5):
    """
    \psi particle filter for multivariate gaussian \psi with linear gaussian transitions. Here adaptive resampling is used

    Parameters
    ----------
    n : int
        number of dimensions
    psi_mean : np.array
        numpy array containing means for the gaussian \psi functions.
    psi_cov : np.array
        numpy array containing covariance matrices for the gaussian \psi functions.
    x_0 : np.array
        Initial guess for the states.
    y : np.array
        Array of observations.
    emission_dens : function
        emission denisty.
    state_sample : np.array
        Array containing parameters for state transition. These are constant a, scaling b, standard deviation s
    N : int, optional
        Number of particles. The default is 1000.
    seed : float, optional
        Seed to be used. The default is 123.
    res_rate : float, optional
        Resampling rate. The default is 0.5.


    Returns
    -------
    output : particles
        Particle class object with the output of the particle filter.

    """ 
    np.random.seed(seed)
    nT = len(y)
    x_sim = np.empty(shape=(nT,N,n))
    w_mat = np.empty(shape=(nT,N))
    Neff = np.empty(shape = nT)
    res_count = 0
    w_mean = np.zeros(shape = (nT,N))+1
    
    a, b, s = state_sample
    m = a + b@x_0
    s_twisted = np.linalg.inv(np.linalg.inv(s)+ np.linalg.inv(psi_cov[0]))
    m_twisted = s_twisted@(np.linalg.inv(s)@np.transpose(m) + np.linalg.inv(psi_cov[0])@np.transpose(psi_mean[0,]))
    x_sim[0,] = np.random.multivariate_normal(mean = m_twisted, cov = s_twisted, size = N)

    for j in range(N):
        u = np.log(emission_dens(y[0], x_sim[0,j,]))
        sc = log_psi_tilde_t(x_sim[0,j,],a,b,s,psi_mean[1,], psi_cov[1])+log_psi_tilde_t(x_0,a,b,s,psi_mean[0,], psi_cov[0])-multivariate_normal.logpdf(x_sim[0,j,], mean = psi_mean[0,], cov = psi_cov[0])
        w_mat[0,j] = np.exp(u+sc)
        
        
    Neff[0] = sum(w_mat[0,])**2/sum(w_mat[0,]**2)
    if Neff[0] < res_rate*N:
        w_mean[0,] = w_mat[0,]
        
    for i in range(1,nT):    
        
        s_twisted = np.linalg.inv(np.linalg.inv(s)+ np.linalg.inv(psi_cov[i]))
        
        if(Neff[i-1] < res_rate*N):
            res_count += 1
            for j in range(N):
                ind = np.random.choice(range(N), size = 1, p = w_mat[i-1,]/sum(w_mat[i-1,]))[0]
                m = a + b@x_sim[i-1,ind,]
                m_twisted = s_twisted@(np.linalg.inv(s)@np.transpose(m) + np.linalg.inv(psi_cov[i])@np.transpose(psi_mean[i,]))
                x_sim[i,j,] = np.random.multivariate_normal(mean = m_twisted, cov = s_twisted, size = 1) 
                if i != (nT-1):
                    w = np.log(emission_dens(y[i], x_sim[i,j,]))
                    sc = log_psi_tilde_t(x_sim[i,j,], a, b, s, psi_mean[i+1,], psi_cov[i+1])-multivariate_normal.logpdf(x_sim[i,j,], mean = psi_mean[i,], cov = psi_cov[i])
                    w_mat[i,j] = np.exp(w+sc)
                else:
                    w = np.log(emission_dens(y[i], x_sim[i,j,]))                    
                    sc = -multivariate_normal.logpdf(x_sim[i,j,], mean = psi_mean[i,], cov = psi_cov[i])
                    w_mat[i,j] = np.exp(w+sc)
                    
                
        else:
            for j in range(N):
                m = a + b@x_sim[i-1,j,]
                m_twisted = s_twisted@(np.linalg.inv(s)@np.transpose(m) + np.linalg.inv(psi_cov[i])@np.transpose(psi_mean[i,]))

                x_sim[i,j,] = np.random.multivariate_normal(mean = m_twisted, cov = s_twisted, size = 1) 
                    
                if i != (nT-1):
                    w = np.log(emission_dens(y[i], x_sim[i,j,]))
                    sc = log_psi_tilde_t(x_sim[i,j,], a, b, s, psi_mean[i+1,], psi_cov[i+1])-multivariate_normal.logpdf(x_sim[i,j,], mean = psi_mean[i,], cov = psi_cov[i])
                    w_mat[i,j] = w_mat[i-1,j]*np.exp(w+sc)
                else:
                    w = np.log(emission_dens(y[i], x_sim[i,j,]))                    
                    sc = -multivariate_normal.logpdf(x_sim[i,j,], mean = psi_mean[i,], cov = psi_cov[i])
                    w_mat[i,j] = w_mat[i-1,j]*np.exp(w+sc)
        
        
        if i == (nT-1):
            w_mean[i,] = w_mat[i,]

        Neff[i] = sum(w_mat[i,])**2/sum(w_mat[i,]**2)
        if Neff[i] < res_rate*N:
            w_mean[i,] = w_mat[i,]

        
    output = particles(x_mat = x_sim, w_mat = w_mat, Neff = Neff, params=None, res_count = res_count, lik = w_mean)
    return output


