from particles import particles
import numpy as np

def bpf_gaussian(n, x_0, y, emission_dens, state_sample, N = 1000, seed = 123, resample = True, res_rate = 0.5, filter_out = True,):
    """
    Bootstrap particle filter where the transition densities are identical normal distributions and the state space is multi dimensional.
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
    x_sim = np.empty(shape=(nT,N,n))
    w_mat = np.empty(shape=(nT,N))
    Neff = np.empty(shape = nT)
    w_mean = np.zeros(shape = (nT,N))+1
    
    res_count = 0
    a, b, s = state_sample
    m = a + b@x_0
    x_sim[0,] = np.random.multivariate_normal(mean = m, cov = s, size = N)
    for j in range(N):
        w = emission_dens(y[0], x_sim[0,j,])
        w_mat[0,j] = w

    
    Neff[0] = sum(w_mat[0,])**2/sum(w_mat[0,]**2)
    if Neff[0] < res_rate*N:
        w_mean[0,] = w_mat[0,]
    
    for i in range(1,nT):    
        #print(i)
        if(Neff[i-1] < res_rate*N):
            res_count += 1
            for j in range(N):
                ind = np.random.choice(range(N), size = 1, p = w_mat[i-1,]/sum(w_mat[i-1,]))[0]
                x_sim[i,j,] = np.random.multivariate_normal(mean = a+b@x_sim[i-1,ind,], cov = s, size = 1) 
                    
                w = emission_dens(y[i], x_sim[i,j,])
                w_mat[i,j] = w
                
                
        else:
            for j in range(N):
                
                x_sim[i,j,] = np.random.multivariate_normal(mean = a+b@x_sim[i-1,j,], cov = s, size = 1)
            
                u = emission_dens(y[i], x_sim[i,j,])
                w = w_mat[i-1,j]*u                
                
                w_mat[i,j] = w
        

        if i == (nT-1):
            w_mean[i,] = w_mat[i,]
        #w_mat[i,] = w
        Neff[i] = sum(w_mat[i,])**2/sum(w_mat[i,]**2)
        if Neff[i] < res_rate*N:
            w_mean[i,] = w_mat[i,]

        
    output = particles(x_mat = x_sim, w_mat = w_mat, Neff = Neff, params=None, res_count = res_count, lik = w_mean)
 
    return output

def bpf_gaussian_mul(n, x_0, y, emission_dens, state_sample, N = 1000, seed = 123, resample = True, res_rate = 0.5, filter_out = True,):
    """
    Bootstrap particle filter where the transition densities are identical normal distributions and the state space is multi dimensional.
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
    x_sim = np.empty(shape=(nT,N,n))
    w_mat = np.empty(shape=(nT,N))
    x_filter = np.empty(shape=(nT,N,n))
    w_filter = np.empty(shape=(nT,N))
    Neff = np.empty(shape = nT)
    w_mean = np.zeros(shape = (nT,N))+1
    
    res_count = 0
    a, b, s = state_sample
    m = a + b@x_0
    x_sim[0,] = np.random.multivariate_normal(mean = m, cov = s, size = N)
    for j in range(N):
        w = emission_dens(y[0], x_sim[0,j,])
        w_mat[0,j] = w
    
    w_mat[0,] = w_mat[0,]/sum(w_mat[0,])
    
    x_filter[0,] = x_sim[0,]
    w_filter[0,] = w_mat[0,]

    Neff[0] = sum(w_mat[0,])**2/sum(w_mat[0,]**2)
    if Neff[0] < res_rate*N:
        w_mean[0,] = w_mat[0,]
    
    for i in range(1,nT):    
        for j in range(N):
                
                x_sim[i,j,] = np.random.multivariate_normal(mean = a+b@x_sim[i-1,j,], cov = s, size = 1)
            
                u = emission_dens(y[i], x_sim[i,j,])
                w = w_mat[i-1,j]*u                
                
                w_mat[i,j] = w
                
        w_mat[i,] = w_mat[i,]/sum(w_mat[i,])
        x_filter[i,] = x_sim[i,]
        w_filter[i,] = w_mat[i,]
        
        Neff[i] = sum(w_mat[i,])**2/sum(w_mat[i,]**2)
        if Neff[i] < res_rate*N:
            res_count = res_count + 1
            x_sim, w_mat = resample_fun(x_sim, w_mat, N ,i)
            w = w_mat[i,]
        
    output = particles(x_mat = x_sim, w_mat = w_mat, Neff = Neff, params=None, res_count = res_count, lik = w_mean, x_filter = x_filter, w_filter= w_filter)
 
    return output

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