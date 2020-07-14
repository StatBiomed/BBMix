# Variational inference of beta-binomial mixture model
"""
Variational Inference for beta-binomial mixture model
Initial contribution: Yuanhua Huang
"""
import warnings
import numpy as np
from scipy.stats import entropy
from scipy.sparse import isspmatrix, csc_matrix
from scipy.optimize import minimize
from scipy.special import logsumexp, digamma, betaln
from .statistics import dirichlet_entropy, logbincoeff, normalize, normalize_exp

from .binomial_mix_vb import BinomialMixVB

class BetaBinMixVB():
    """BetaBinMixVB model: variational inference for beta-binomial mixture model
    The prior can be set via set_prior() before fitting the model.
    """
    def __init__(self, n_obs, n_var=1, n_components=1, per_var=False, 
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

    def set_init(self, theta_mu=None, theta_sum=None, pi_size=None, Z=None,
            theta_mu_hyper=None, theta_sum_hyper=None):
        """Set initial values of the parameters to fit

        Key properties
        --------------
        theta_mu: numpy.array (n_var, n_obs)
            Mean of Beta distribution as theta's posterior
        theta_sum: numpy.array (n_var, n_obs)
            Concetration Beta distribution as theta's posterior
        Z: numpy.array (n_obs, n_components)
            Assignment probability of each observation to each cluster
        pi_size: numpy.array (n_components)
            Dirichlet parameters for posterior of cluster proportion
        theta_mu_hyper: numpy.array (n_var, n_components)
            Hyper-parameter mean in Beta distribution as theta's prior
        theta_sum_hyper: numpy.array (n_var, n_components)
            Hyper-parameter concetration in Beta distribution as theta's prior
        """
        # initial key parameters
        if theta_mu is not None:
            self.theta_mu = theta_mu
        else:
            self.theta_mu = np.random.uniform(size=(self.n_var, self.n_obs))

        if theta_sum is not None:
            self.theta_sum = theta_sum
        else:
            self.theta_sum = np.random.uniform(3, 10, size=self.theta_mu.shape)

        if Z is not None:
            self.Z_loglik = np.log(normalize(Z))
        else:
            self.Z_loglik = np.random.uniform(0.5, 1, size=(self.n_obs, self.K))

        if pi_size is not None:
            self.pi_size = pi_size
        else:        
            self.pi_size = np.ones((self.K))

        if theta_mu_hyper is not None:
            _theta_mu_hyper = theta_mu_hyper
        else:
            _theta_mu_hyper = np.random.uniform(
                0.3, 0.7, size=(self.n_var, self.K))
            # self.theta_mu_hyper = np.ones((self.n_var, self.K)) * 0.5
        if theta_sum_hyper is not None:
            _theta_sum_hyper = theta_sum_hyper
        else:
            _theta_sum_hyper = np.ones(_theta_mu_hyper.shape) * 2

        self.theta_s1_hyper = _theta_mu_hyper * _theta_sum_hyper
        self.theta_s2_hyper = (1 - _theta_mu_hyper) * _theta_sum_hyper
           
    
    def set_prior(self, pi_prior=None):
        """Set parameters of prior distribution for: pi. 
        The prior parameters are in the form of its according posterior
        """
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
    def theta_mu_hyper(self):
        return (
            self.theta_s1_hyper / (self.theta_s1_hyper + self.theta_s2_hyper)
        )

    @property
    def theta_sum_hyper(self):
        return self.theta_s1_hyper + self.theta_s2_hyper

    @property
    def pi(self):
        return normalize(self.pi_size)
    
    @property
    def Z(self):
        return normalize_exp(self.Z_loglik)
    
    def update_theta_size(self, AD, DP):
        """Coordinate ascent for updating theta posterior parameters
        """
        BD = DP - AD
        _theta_s1 = np.dot(self.theta_s1_hyper, self.Z.T) + AD
        _theta_s2 = np.dot(self.theta_s2_hyper, self.Z.T) + BD

        self.theta_mu = _theta_s1 / (_theta_s1 + _theta_s2)
        if self.fix_theta_sum == False:
            self.theta_sum = _theta_s1 + _theta_s2
        
    def update_pi_size(self):
        """Coordinate ascent for updating pi posterior parameters
        """
        self.pi_size = self.pi_prior + np.sum(self.Z, axis=0)

    def update_Z_loglik(self, theta_s1_hyper=None, theta_s2_hyper=None):
        """Coordinate ascent for updating assignment probability
        """
        self.Z_loglik = (
            self.get_E_beta(np.append(theta_s1_hyper, theta_s2_hyper)) + 
            np.tile(np.log(self.pi), (self.n_obs, 1))
        )

    def get_E_binom(self, AD, DP):
        """Calculating expected log likelihood of data over theta
        """
        # if logbincoeff_ is None:
        #     logbincoeff_ = logbincoeff(DP, AD)
        BD = DP - AD
        self.binom_E_ = np.sum(
            AD * digamma(self.theta_s1) + 
            BD * digamma(self.theta_s2) - 
            DP * digamma(self.theta_s1 + self.theta_s2), 
            axis=0, keepdims=True
        )

    def get_E_beta(self, params):
        """Calulating expected log likelihood of beta prior over theta, namely
        cross entropy
        """
        theta_s1_hyper = params[:self.n_var * self.K].reshape(-1, self.K)
        theta_s2_hyper = params[self.n_var * self.K:].reshape(-1, self.K)

        _beta_loglik = (
            -np.sum(betaln(theta_s1_hyper, theta_s2_hyper), 0, keepdims=True) +
            np.dot(digamma(self.theta_s1.T), theta_s1_hyper) + 
            np.dot(digamma(self.theta_s2.T), theta_s2_hyper) - 
            np.dot(digamma(self.theta_sum.T), (theta_s1_hyper + theta_s2_hyper))
        )
        return _beta_loglik

    def get_ELBO(self, logbincoeff_=0):
        """Calculating variational evidence lower bound with current parameters
        """
        log_p_dat_theta = logbincoeff_ + np.sum(self.binom_E_)
        log_p_theta_Z = np.sum(self.Z * self.get_E_beta(
            np.append(self.theta_s1_hyper, self.theta_s2_hyper)))
        
        log_q_theta = -np.sum(dirichlet_entropy(
            np.stack((self.theta_s1, self.theta_s2), axis=2)))

        KL_Z, KL_pi = 0, 0
        if self.K > 1:
            KL_Z  = np.sum(entropy(
                self.Z, np.tile(np.log(self.pi), (self.n_obs, 1)), axis=-1))
            KL_pi = np.sum(dirichlet_entropy(
                self.pi_size, self.pi_prior, axis=-1))
        
        return log_p_dat_theta + log_p_theta_Z - log_q_theta - KL_Z - KL_pi

    def maximumization(self, optimize_steps=100):
        """maximumizing ELBO with regards to hyper parameters
        """
        loss_func = lambda x: -np.sum(self.Z * self.get_E_beta(x))
        
        _bounds = tuple([(0.001, 10000)] * (2 * self.K * self.n_var))
        _init_params = np.append(self.theta_s1_hyper, self.theta_s2_hyper)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(loss_func, x0=_init_params,
                           bounds=_bounds,
                           method='SLSQP',
                           options={'disp': False,
                                    'maxiter': optimize_steps}
                           )

        _len = self.K * self.n_var
        self.theta_s1_hyper[:, :] = res.x[:_len].reshape(-1, self.K)
        self.theta_s2_hyper[:, :] = res.x[_len:].reshape(-1, self.K)
        return res
    

    def binomial_init(self, AD, DP):
        """Warm initialization of parameters
        """
        models_all = []
        for i in range(50):
            _model = BinomialMixVB(self.n_obs, self.n_var, self.K)
            _model.fit(csc_matrix(AD), csc_matrix(DP), min_iter=20)
            models_all.append(_model)
        _model = models_all[np.argmax([x.ELBO_[-1] for x in models_all])]

        _theta_mu_hyper  = _model.theta_mu
        _theta_sum_hyper = np.ones(_theta_mu_hyper.shape) * 15
        self.Z_loglik = _model.Z_loglik

        self.theta_s1_hyper = _theta_mu_hyper * _theta_sum_hyper
        self.theta_s2_hyper = (1 - _theta_mu_hyper) * _theta_sum_hyper


    def fit_VB(self, AD, DP, n_iter=5):
        """Fit VB given hyper parameters
        """
        _ELBO = np.zeros(n_iter)
        for it in range(n_iter):
            self.update_theta_size(AD, DP)
            self.update_Z_loglik(self.theta_s1_hyper, self.theta_s2_hyper)
            self.get_E_binom(AD, DP)
            
            if self.fix_pi == False:
                self.update_pi_size()
            
            _ELBO[it] = self.get_ELBO(self.logbincoeff_)

        return _ELBO

    def fit(self, AD, DP, init_method='binomial', n_init=50, 
        max_iter=200, min_iter=5, decreasing_stop=True,
        epsilon_conv=1e-2, verbose=True):
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
        init_method : str
            Method to initalise the clustering, default='binomial' for 
            BinomialMixVB model
        n_init : int
            Number of initials used in the init_method
        max_iter : int
            Maximum number of iterations
        min_iter :
            Minimum number of iterations
        epsilon_conv : float
            Threshold for detecting convergence
        verbose : bool
            Whether print out log info
        """
        # check AD and DP's type: scipy.sparse is not supported yet
        if isspmatrix(AD) or isspmatrix(DP):
            AD = AD.toarray()
            DP = DP.toarray()

        self.logbincoeff_ = np.sum(logbincoeff(
            np.array(DP[DP>0]).reshape(-1), 
            np.array(AD[DP>0]).reshape(-1)
        ))

        # warm initialization
        if init_method == 'binomial':
            self.binomial_init(AD, DP)
        
        ELBO = np.zeros(max_iter)
        for it in range(max_iter):
            _elbo = self.fit_VB(AD, DP, n_iter=5)
            _ = self.maximumization()

            ELBO[it] = _elbo[-1]
            if it > min_iter:
                if (ELBO[it] - ELBO[it - 1]) < -epsilon_conv:
                    if decreasing_stop:
                        print("Warning: stopped due to ELBO decreases! " +
                                "%.2f to %.2f!" %(ELBO[it - 1], ELBO[it]))
                    else:
                        if verbose:
                            print("Warning: stopped due to ELBO decreases! " +
                                  "%.2f to %.2f!" %(ELBO[it - 1], ELBO[it]))                    
                elif it == max_iter - 1:
                    if verbose:
                        print("Warning: VB did not converge!\n")
                elif ELBO[it] - ELBO[it - 1] < epsilon_conv:
                    break
        
        self.ELBO_ = np.append(self.ELBO_, ELBO[:it])
