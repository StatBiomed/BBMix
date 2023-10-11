# -*- coding: utf-8 -*-
"""binomial_mix.

Chen Qiao: cqiao@connect.hku.hk
"""

import sys
import warnings

import numpy as np
from scipy.special import gammaln, logsumexp

from .model_base import ModelBase


class MixtureBinomial(ModelBase):
    """Mixture of Binomial Models

    This class implements EM algorithm for parameter estimation of Mixture 
    of Binomial models. 

    Attributes:
        n_components (int): number of mixtures.
        tor (float): tolarance difference for earlier stop training.
        params (numpy float array): parameters of the model, [p_1, p_2, ..., 
            p_K, pi_1, pi_2, ..., pi_K],
            None before parameter estimation.
        losses (list): list of negative loglikelihood losses of the training 
            process, None before parameter estimation.
        model_scores (dict): scores for the model, including "BIC" and "ICL" scores

    Notes
    -----
    Because M-step has analytical solution, parameter estimation is fast.

    Usage:

        em_mb = MixtureBinomial(
        n_components=2,
        tor=1e-6)
        params = em_mb.EM((ys, ns), max_iters=250, early_stop=True)

    Simulation experiment:

        import numpy as np
        from scipy.stats import bernoulli, binom
        from bbmix.models import MixtureBinomial
        n_samples = 2000
        n_trials = 1000
        pis = [0.6, 0.4]
        p1, p2 = 0.4, 0.8
    
        gammars = bernoulli.rvs(pis[0], size=n_samples)
        n_pos_events = sum(gammars)
        n_neg_events = n_samples - n_pos_events
    
        ys_of_type1 = binom.rvs(n_trials, p1, size=n_pos_events)
        ys_of_type2 = binom.rvs(n_trials, p2, size=n_neg_events)
    
        ys = np.concatenate((ys_of_type1, ys_of_type2))
        ns = np.ones(n_samples, dtype=np.int) * n_trials
    
        em_mb = MixtureBinomial(
            n_components=2,
            tor=1e-20)
    
        params = em_mb.fit((ys, ns), max_iters=250, early_stop=True)
        print(params)
        print(p1, p2, pis)
        print(em_mb.model_scores)

    """

    def __init__(self,
                 n_components=2,
                 tor=1e-6
                 ):
        """Initialization method

        Args:
            n_components (int): number of mixtures. Defaults to 2.
            tor (float): tolerance shreshold for early-stop training. 
                Defaults to 1e-6.

        """
        super(MixtureBinomial, self).__init__()
        self.n_components = n_components
        self.tor = tor

    def E_step(self, y, n, params):
        """Expectation step

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            params (np.array): model parameters 

        Returns:
            np.array: expectation of the latent variables
        """
        E_gammas = [None] * self.n_components
        for k in range(self.n_components):
            p_k, pi_k = params[k], params[k + self.n_components]
            E_gammas[k] = y * np.log(p_k) + (n - y) * \
                          np.log(1 - p_k) + np.log(pi_k)

        # normalize as they havn't been
        E_gammas = E_gammas - logsumexp(E_gammas, axis=0)
        return np.exp(E_gammas)

    def M_step(self, y, n, E_gammas, params):
        """Maximization step

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            E_gammas (np.array): results of E step
            params (np.array): model parameters 

        Returns:
            np.array: updated model parameters
        """
        N_samples = len(n)
        for k in range(self.n_components):
            E_gammas[k][E_gammas[k] == 0] = 1e-20
            params[k] = np.sum(y * E_gammas[k]) / np.sum(n * E_gammas[k])
            params[k + self.n_components] = np.sum(E_gammas[k]) / N_samples
        return params

    def log_likelihood_binomial(self, y, n, p, pi=1.0):
        """log likelihood of data under binomial distribution

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            p (float): probability of positive event
            pi (float): weight of mixture component

        Returns:
            np.array: log likelihood of data
        """
        return gammaln(n + 1) - (gammaln(y + 1) + gammaln(n - y + 1)) \
               + y * np.log(p) + (n - y) * np.log(1 - p) + np.log(pi)

    def log_likelihood_mixture_bin(self, y, n, params):
        """log likelihood of dataset under mixture of binomial distribution

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            params (np.array): parameters of the model

        Returns:
            float: log likelihood of the dataset
        """
        logLik_mat = np.zeros((len(n), self.n_components), dtype=float)
        for k in range(self.n_components):
            p_k, pi_k = params[k], params[k + self.n_components]
            logLik_mat[:, k] = self.log_likelihood_binomial(y, n, p_k, pi_k)
        return logsumexp(logLik_mat, axis=1).sum()

    def EM(self, y, n, params, max_iters=250, early_stop=False, n_tolerance=10,
           verbose=False):
        """EM algorithim

        Args:
            y (np.array): number of positive events
            n (np.array): total number of trials respectively
            params (list): init model params
            max_iters (int, optional): maximum number of iterations for EM. Defaults to 250.
            early_stop (bool, optional): whether early stop training. Defaults to False.
            n_tolerance (int): the max number of violations to trigger early stop.
            pseudocount (float) : add pseudocount if data is zero
            verbose (bool, optional): whether print training information. Defaults to False.

        Returns:
            np.array: trained parameters
        """
        n_tol = n_tolerance
        losses = [sys.maxsize]

        for ith in range(max_iters):

            # E step
            E_gammas = self.E_step(y, n, params)

            # M step
            params = self.M_step(y, n, E_gammas, params)

            # record current NLL loss
            losses.append(-self.log_likelihood_mixture_bin(y, n, params))

            if verbose:
                print("=" * 10, "Iteration {}".format(ith + 1), "=" * 10)
                print("Current params: {}".format(params))
                print("Negative LogLikelihood Loss: {}".format(losses[-1]))
                print("=" * 25)

            improvement = losses[-2] - losses[-1]
            if early_stop:
                if improvement < self.tor:
                    n_tol -= 1
                else:
                    n_tol = n_tolerance
                if n_tol == 0:
                    if verbose:
                        print("Improvement halts, early stop training.")
                    break
             
        self.score_model(len(params), len(y), losses[-1], E_gammas)
        self.params = params
        self.losses = losses[1:]
        return params

    def _param_init(self, y, n):
        """Initialziation of model parameters

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
        Returns:
            np.array: initialized model parameters
        """
        return np.concatenate([np.random.uniform(0.49, 0.51, self.n_components),
                               np.random.uniform(0.4, 0.6, self.n_components)])
                
    def fit(self, data, max_iters=250, early_stop=False, pseudocount=0.1,
            n_tolerance=10, verbose=False):
        """Fit function

        Args:
            data (tuple of arrays): y, n: number of positive events and total number of trials respectively
            max_iters (int, optional): maximum number of iterations for EM. Defaults to 250.
            early_stop (bool, optional): whether early stop training. Defaults to False.
            pseudocount (float) : add pseudocount if data is zero
            n_tolerance (int): the max number of violations to trigger early stop.
            verbose (bool, optional): whether print training information. Defaults to False.

        Returns:
            np.array: trained parameters
        """
        y, n = data
        self.nzero_prop = np.sum(y > 0)/np.shape(y)[0]
        y, n = self._preprocess(data, pseudocount)

        init_params = self._param_init(y, n)
        if verbose:
            print("=" * 25)
            print("Init params: {}".format(init_params))
            print("=" * 25)
        params = self.EM(y, n, init_params, max_iters=max_iters,
                         early_stop=early_stop, verbose=verbose,
                         n_tolerance=n_tolerance)
        if self.n_components == 2 and np.abs(params[0] - params[1]) < 1e-4 and verbose:
            print("Colapsed to one component, please check proportion of non-zero counts.")
        return params

    def sample(self, n_trials):
        """Generate data from fitted parameters
        n_trails :
        Args:
            n_trails (array_like): total number of trials

        Returns:
            np.array: ys generated from the fitted distribution
        """
        if hasattr(self, 'params') == False:
            raise Exception("Error: please fit the model or set params before sample()")

        mus = self.params[:self.n_components]
        pis = self.params[self.n_components: 2 * self.n_components]

        labels = np.random.choice(self.n_components, size=n_trials.shape, p=pis)
        ys_out = np.zeros(n_trials.shape, dtype=int)
        for i in range(self.n_components):
            _idx = np.where(labels == i)
            ys_out[_idx] = binom.rvs(n_trials[_idx].astype(np.int32), mus[i])

        return ys_out


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import bernoulli, binom
    from bbmix.models import MixtureBinomial
    n_samples = 2000
    n_trials = 1000
    pis = [0.6, 0.4]
    p1, p2 = 0.4, 0.8

    gammars = bernoulli.rvs(pis[0], size=n_samples)
    n_pos_events = sum(gammars)
    n_neg_events = n_samples - n_pos_events

    ys_of_type1 = binom.rvs(n_trials, p1, size=n_pos_events)
    ys_of_type2 = binom.rvs(n_trials, p2, size=n_neg_events)

    ys = np.concatenate((ys_of_type1, ys_of_type2))
    ns = np.ones(n_samples, dtype=np.int) * n_trials

    em_mb = MixtureBinomial(
        n_components=2,
        tor=1e-20)

    params = em_mb.fit((ys, ns), max_iters=250, early_stop=True)
    print(params)
    print(p1, p2, pis)
    print(em_mb.model_scores)
