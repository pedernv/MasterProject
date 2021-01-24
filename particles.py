import numpy as np
class particles:
    """
    Class for containing output of bootstrap particle filter and \psi -APF
    """
    def __init__(self, x_mat, w_mat, Neff, x_filter = None, w_filter = None, params = None, res_count = None, lik = None):
        """
        Init for particle class

        Parameters
        ----------
        x_mat : np.array
            Array containing particle estimates. Needs to be weighted using weights from w_mat
        w_mat : np.array
            Array contating weights of particles.
        Neff : np.array
            The effective number of parameters at each step.
        params : np.array
            Parameters of the particle filter used.
        res_count : int, optional
            Number of times resampled. The default is None.
        lik : np.array, optional
            Weights used for estimating the (log)likelihood using adaptive resampling. The default is None.

        Returns
        -------
        None.

        """
        self.x = x_mat
        self.w = w_mat
        self.x_f = x_filter
        self.w_f = w_filter
        self.res = res_count
        self.neff = Neff
        self.params = params
        self.lik = lik
        
    def loglik(self):
        """
        Function for computing the loglikelihood based on a particle filter with adaptive resampling

        Returns
        -------
        ll : float
            Loglikelihood from the particle filter based on the data.

        """
        ll = np.sum(np.log(np.mean(self.lik, axis = 1)))
        return ll
    
    def n_particles(self):
        """
        Gives the number of particles used

        Returns
        -------
        n : int
            The number of particles.

        """
        n = self.x_mat.shape[1]
        return n
    
    def n_obs(self):
        """
        Gives the number of observations.

        Returns
        -------
        n : int
            Number of observatios.

        """
        n = self.x_mat.shape[0]
        return n
        
    def smooth(self):
        pass
        