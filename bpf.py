import numpy as np
import matplotlib.pyplot as plt
from particles import particles

def resample_fun(x_in, w_in, N, i):
    """
    Multinomial resampling on particles x_in with resampling based on normalized weights w_in

    Parameters
    ----------
    x_in : np.array
        numpy array of particle realizations.
    w_in : np.array
        numpy array of weights.
    N : int
        Number of particles.
    i : int
        Current timestep.

    Returns
    -------
    x_out : np.array
        numpy array of resampled particles.
    w_out : np.array
        numpy array of resampled weights with step i being 1/N.

    """
    w_out = w_in
    x_out = x_in
    
    ind = np.random.choice(range(N), size = N, replace = True, p = w_in[i,])
    w = np.array([1/N]*N)
    w_out[i,] = w
    
    for j in range(len(ind)):
        x_out[:,j] = x_in[:,ind[j]]
        w_out[:,j] = w_out[:,ind[j]]
    return x_out, w_out

def bpf_normal(x_0, y, emission_dens, state_sample, N = 10000, seed = 123, resample = True, res_rate = 0.5, filter_out = True):
    """
    Bootstrap particle filter where the transition densities are identical normal distributions and the state space is one dimensional.
    The transition densities are on the form
    
    f(x_t|x_t-1) = N(\cdot; \mu = a+b*x_t-1, sigma^2 = s^2)
    
    where a,b and s are included in the list state_sample
    
    Parameters
    ----------
    x_0 : float
        Initial value of path
    y : float
        observations
    emission_dens : function
        Probability mass/density of the observations given the current state
    state_sample : TYPE
        DESCRIPTION.
    N : int, optional
        Number of particles. The default is 1000.
    seed : TYPE, optional
        seed for reproducability. The default is 123.
    resample : Boolean, optional
        If true resampling is done. The default is True.
    res_rate : float, optional
        Determines how often resampling is done. The default is 0.5.
    filter_out : Boolean, optional
        If true reuturns filtered values of states and weights. The default is True.

    Returns
    -------
    output : particles
        particles object containing the results of the particle filter

    """
    np.random.seed(seed)
    nT = len(y)
    x_sim = np.empty(shape=(nT,N))
    w_mat = np.empty(shape=(nT,N))
    x_filter = None; w_filter = None
    Neff = np.empty(shape = nT)
    
    res_count = 0
    a, b, s = state_sample
    x_sim[0,] = np.random.normal(loc = a + b*x_0, scale = s, size = N)
    
    w = emission_dens(y[0], x_sim[0,])
    w = w/sum(w)
    
    w_mat[0,] = w
    
    if filter_out:
        x_filter = np.empty(shape = (nT,N))
        w_filter = np.empty(shape = (nT,N))
        x_filter[0,] = x_sim[0,]
        w_filter[0,] = w_mat[0,]
    
    Neff[0] = 1/sum(w**2)

    for i in range(1,nT):
        x_sim[i,] = np.random.normal(loc = 0, scale = s, size = N)+(a+b*x_sim[i-1,])
        
        u = emission_dens(y[i], x_sim[i,])
        w = u*w_mat[i-1,]
        w = w/sum(w)
        w_mat[i,] = w
        Neff[i] = 1/sum(w**2)
        
        if resample:
            if Neff[i] < res_rate*N:
                res_count = res_count + 1
                x_sim, w_mat = resample_fun(x_sim, w_mat, N ,i)
                w = w_mat[i,]
                
        if filter_out:
            x_filter[i,] = x_sim[i,]
            w_filter[i,] = w_mat[i,]

    output = particles(x_mat = x_sim, w_mat = w_mat, params = state_sample, res_count = res_count, Neff = Neff, w_filter = w_filter, x_filter = x_filter)
    
    return output

def bpf_normal_alt(x_0, y, emission_dens, state_sample, N = 1000, seed = 123, resample = True, res_rate = 0.5):
    """
    Bootstrap particle filter where the transition densities are identical normal distributions and the state space is one dimensional.
    The transition densities are on the form
    
    f(x_t|x_t-1) = N(\cdot; \mu = a+b*x_t-1, sigma^2 = s^2)
    
    where a,b and s are included in the list state_sample
    
    Parameters
    ----------
    x_0 : float
        Initial value of path
    y : float
        observations
    emission_dens : function
        Probability mass/density of the observations given the current state
    state_sample : TYPE
        DESCRIPTION.
    N : int, optional
        Number of particles. The default is 1000.
    seed : TYPE, optional
        seed for reproducability. The default is 123.
    resample : Boolean, optional
        If true resampling is done. The default is True.
    res_rate : float, optional
        Determines how often resampling is done. The default is 0.5.
    filter_out : Boolean, optional
        If true reuturns filtered values of states and weights. The default is True.

    Returns
    -------
    output : particles
        particles object containing the results of the particle filter

    """
    np.random.seed(seed)
    nT = len(y)
    x_sim = np.empty(shape=(nT,N))
    w_mat = np.empty(shape=(nT,N))
    x_filter = np.empty(shape=(nT,N))
    w_filter = np.empty(shape=(nT,N))
    Neff = np.empty(shape = nT)
    w_mean = np.zeros(shape = (nT,N))+1
    
    res_count = 0
    a, b, s = state_sample
    x_sim[0,] = np.random.normal(loc = a + b*x_0, scale = s, size = N)
    
    w = emission_dens(y[0], x_sim[0,])
    w_mat[0,] = w
    x_filter[0,] = x_sim[0,]
    w_filter[0,] = w_mat[0,]
    Neff[0] = sum(w)**2/sum(w**2)
    if Neff[0] < res_rate*N:
        w_mean[0,] = w_mat[0,]
    
    for i in range(1,nT):        
        if(Neff[i-1] < res_rate*N):
            res_count += 1
            for j in range(N):
                ind = np.random.choice(range(N), size = 1, p = w_mat[i-1,]/sum(w_mat[i-1,]))
                x_sim[i,j] = np.random.normal(loc = 0, scale = s, size = 1)[0] +(a+b*x_sim[i-1,ind])
                    
                w = emission_dens(y[i], x_sim[i,j])
                w_mat[i,j] = w
            x_filter[i,] = x_sim[i,]
            w_filter[i,] = w_mat[i,] 
                
        else:
            x_sim[i,] = np.random.normal(loc = 0, scale = s, size = N)+(a+b*x_sim[i-1,])
            
            u = emission_dens(y[i], x_sim[i,])
            w = w_mat[i-1,]*u                
                
            w_mat[i,] = w
            x_filter[i,] = x_sim[i,]
            w_filter[i,] = u
        

        if i == (nT-1):
            w_mean[i,] = w_mat[i,]
            
        Neff[i] = sum(w_mat[i,])**2/sum(w_mat[i,]**2)
        if Neff[i] < res_rate*N:
            w_mean[i,] = w_mat[i,]
        
    output = particles(x_mat = x_sim, w_mat = w_mat, Neff = Neff, params=None, res_count = res_count, lik = w_mean, w_filter = w_filter, x_filter = x_filter)
 
    return output



def plot_pf(par, ylab = "Time", xlab = "", conf = False, m = False, val = [False, 0,0]):
    x = np.diag(np.dot(par.x_f, np.transpose(par.w_f)))
    if conf:
        c = np.quantile(par.x_f, (0.025, 0.975), axis = 1)
        plt.plot(x)
        plt.fill_between(x, c[0], c[1], color='b', alpha=.1)
        #plt.plot(c[0])
        #plt.plot(c[1])
        
    if m:
        plt.plot(np.max(par.x_f, axis = 1))
        plt.plot(np.min(par.x_f, axis = 1))
    if val[0]:
        plt.plot(x[val[1]:val[2]])
        return
    plt.plot(np.linspace(1,50),x)
    plt.xlabel(ylab)
    plt.ylabel(xlab)

def plot_pf_ada(par, ylab = "Time", xlab = "", conf = True):
    n = len(par.x)
    x = np.empty(shape = n)
    for i in range(n):
        x[i] = np.mean(par.x[i,])
    
    plt.plot(x)
    plt.xlabel(ylab)
    plt.ylabel(xlab)