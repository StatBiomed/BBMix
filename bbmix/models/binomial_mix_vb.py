# Variational inference of binomial mixture model
"""
Variational Inference for binomial mixture model
Initial development: Yuanhua Huang
"""

import numpy as np
from scipy.stats import entropy
from scipy.sparse import isspmatrix
from scipy.optimize import minimize
from scipy.special import logsumexp, digamma
from .statistics import dirichlet_entropy, logbincoeff, normalize, normalize_exp

class BinomialMixVB():
    """BinMixVB model: Variational Inference for binomial mixture model
    The prior can be set via set_prior() before fitting the model.
    """
    def __init__(self, n_obs, n_var=1, n_components=1, 
        fix_theta_sum=False, fix_pi=False):
        """Initialise Vireo model
        Note, multiple initializations are highly recomended to avoid local 
        optima.
        
        Parameters
        ----------
        n_obs : int 
            Number of observations, i.e., samples
        n_var : int
            Number of variables, i.e., features or dimensions
        n_components : int
            Number of mixture components
        fix_theta_sum: bool
            Whether fix the concetration parameter of theta's posterior
        fix_pi: bool
            Whether fix the pi to the initial value
        """
        self.n_var = n_var
        self.n_obs = n_obs
        self.n_components = n_components
        self.K = self.n_components # for short cut

        self.fix_pi = fix_pi
        self.fix_theta_sum = fix_theta_sum
        
        # initial values by random initial
        self.set_init()
        self.set_prior()
        self.ELBO_ = np.zeros((0))

    def set_init(self, theta_mu=None, theta_sum=None, pi_size=None, Z=None):
        """Set initial values of the parameters to fit

        Key properties
        --------------
        theta_mu: numpy.array (n_var, n_components)
            Mean of Beta distribution as theta's posterior
        theta_sum: numpy.array (n_var, n_components)
            Concetration Beta distribution as theta's posterior
        Z: numpy.array (n_obs, n_components)
            Assignment probability of each observation to each cluster
        pi_size: numpy.array (n_components)
            Dirichlet parameters for posterior of cluster proportion
        """
        # initial key parameters
        if theta_mu is not None:
            self.theta_mu = theta_mu
        else:
            self.theta_mu = np.random.uniform(size=(self.n_var, self.K))

        if theta_sum is not None:
            self.theta_sum = theta_sum
        else:
            self.theta_sum = np.random.uniform(3, 10, size=(self.n_var, self.K))

        if Z is not None:
            self.Z_loglik = np.log(normalize(Z))
        else:
            self.Z_loglik = np.random.uniform(0.5, 1, size=(self.n_obs, self.K))

        if pi_size is not None:
            self.pi_size = pi_size
        else:        
            self.pi_size = np.ones((self.K))
    
    
    def set_prior(self, theta_mu_prior=None, theta_sum_prior=None, 
                  pi_prior=None):
        """Set parameters of prior distribution for key variables:theta, pi, Z. 
        The prior parameters are in the form of its according posterior
        """
        if theta_mu_prior is None:
            theta_mu_prior = np.ones((self.n_var, self.K)) * 0.5
        if theta_sum_prior is None:
            theta_sum_prior = np.ones(theta_mu_prior.shape) * 2
        self.theta_s1_prior = theta_mu_prior * theta_sum_prior
        self.theta_s2_prior = (1 - theta_mu_prior) * theta_sum_prior

        if pi_prior is not None:
            self.pi_prior = pi_prior
        else:
            self.pi_prior = np.ones(self.K)

    @property
    def theta_s1(self):
        """Beta concetration1 parameter for theta posterior"""
        return self.theta_mu * self.theta_sum

    @property
    def theta_s2(self):
        """Beta concetration2 parameter for theta posterior"""
        return (1 - self.theta_mu) * self.theta_sum

    @property
    def pi(self):
        return normalize(self.pi_size)
    
    @property
    def Z(self):
        return normalize_exp(self.Z_loglik)

    @property
    def Z_prior(self):
        return np.tile(self.pi, (self.n_obs, 1))
    
    def update_theta_size(self, AD, DP):
        """Coordinate ascent for updating theta posterior parameters
        """
        BD = DP - AD
        _theta_s1 = self.theta_s1_prior + AD * self.Z
        _theta_s2 = self.theta_s2_prior + BD * self.Z

        self.theta_mu = _theta_s1 / (_theta_s1 + _theta_s2)
        if self.fix_theta_sum == False:
            self.theta_sum = _theta_s1 + _theta_s2
        
    def update_pi_size(self):
        """Coordinate ascent for updating pi posterior parameters
        """
        self.pi_size = self.pi_prior + np.sum(self.Z, axis=0)

    def update_Z_loglik(self, AD, DP):
        """Coordinate ascent for updating assignment probability
        """
        BD = DP - AD
        self.Z_loglik = (
            np.log(self.Z_prior) + 
            AD.T * digamma(self.theta_s1) + 
            BD.T * digamma(self.theta_s2) - 
            DP.T * digamma(self.theta_s1 + self.theta_s2)
        )
        
    def get_ELBO(self, logbincoeff_=0):
        """Calculating variational evidence lower bound with current parameters
        """
        _logLik_E = self.Z_loglik -  np.log(self.Z_prior)

        LB_lik = np.sum(_logLik_E * self.Z)
        KL_Z   = np.sum(entropy(self.Z, self.Z_prior, axis=-1))
        KL_pi  = np.sum(dirichlet_entropy(self.pi_size, self.pi_prior, axis=-1))
        KL_theta = np.sum(dirichlet_entropy(
            np.stack((self.theta_s1, self.theta_s2), axis=2),
            np.stack((self.theta_s1_prior, self.theta_s2_prior), axis=2)
        ))

        return logbincoeff_ + LB_lik - KL_Z - KL_pi - KL_theta
    

    def fit(self, AD, DP, max_iter=200, min_iter=5, epsilon_conv=1e-2,
        verbose=True):
        """Fit Vireo model with coordinate ascent
        For high dimentional data or high number of components, please run 
        with multiple initializations, e.g., 100 to avoid local optima, and
        pick the one with highest ELBO.

        Parameters
        ----------
        AD : scipy.sparse.csc_matrix (n_var, n_obs)
            Sparse count matrix for alternative events or success
        DP : scipy.sparse.csc_matrix (n_var, n_obs)
            Sparse count matrix for depths or total trials
        max_iter : int
            Maximum number of iterations
        min_iter :
            Minimum number of iterations
        epsilon_conv : float
            Threshold for detecting convergence
        verbose : bool
            Whether print out log info
        """
        # check AD and DP's type: scipy.sparse or numpy.matrix
        if isspmatrix(AD) == False or isspmatrix(DP) == False:
            AD = np.asmatrix(AD)
            DP = np.asmatrix(DP)

        self.logbincoeff_ = np.sum(logbincoeff(
            np.array(DP[DP>0]).reshape(-1), 
            np.array(AD[DP>0]).reshape(-1)
        ))
        
        ELBO = np.zeros(max_iter)
        for it in range(max_iter):
            self.update_Z_loglik(AD, DP)
            self.update_theta_size(AD, DP)
            if ~self.fix_pi: self.update_pi_size()

            ELBO[it] = self.get_ELBO(self.logbincoeff_)

            if it > min_iter:
                if (ELBO[it] - ELBO[it - 1]) < -epsilon_conv:
                    if verbose:
                        print("Warning: Lower bound decreases! %.2f to %.2f!\n"
                              %(ELBO[it - 1], ELBO[it]))
                elif it == max_iter - 1:
                    if verbose:
                        print("Warning: VB did not converge!\n")
                elif ELBO[it] - ELBO[it - 1] < epsilon_conv:
                    break
        
        self.ELBO_ = np.append(self.ELBO_, ELBO[:it])
