# -*- coding: utf-8 -*-
"""betabin_mix.

Chen Qiao: cqiao@connect.hku.hk
"""

import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import beta, gammaln, logsumexp
from scipy.stats import bernoulli, betabinom
from sklearn.cluster import KMeans

from .binomial_mix import MixtureBinomial
from .model_base import ModelBase


class MixtureBetaBinomial(ModelBase):
    """Mixture of Beta-Binomial Models

    This class implements EM algorithm for parameter estimation of Mixture 
    of Beta-Binomial models. 

    Attributes:
        n_components (int): number of mixtures.
        max_m_step_iter (int): maximum number of iterations for sub-optimization at M step
        tor (float): tolarance difference for earlier stop training.
        params (numpy float array): parameters of the model, [alph_1, alpha_2, ..., 
            beta_1, beta_2, ..., pi_1, pi_2, ..., pi_K],
            None before parameter estimation.
        losses (list): list of negative loglikelihood losses of the training 
            process, None before parameter estimation.
        model_scores (dict): scores for the model, including "BIC" and "ICL" scores

    Notes
    -----
    Because M-step has no analytical solution, used Scipy solvel 'SLSQP' for solving a 
    a constrained optimization problem for NLL loss of the dataset.

    To fasten convergence, provided warm initialization methods for model parameters:
        - k-means
        - mixture of binomial distributions

    Usage:

        em_mbb = MixtureBetaBinomial(
        n_components=2,
        max_m_step_iter=250,
        tor=1e-6)

        params = em_mbb.EM((ys, ns), max_iters=200, warm_init='mixbin', early_stop=False)

    Simulation experiment:

        from scipy.stats import bernoulli, binom, betabinom

        n_samples = 2000
        n_trials = 1000
        pis = [0.6, 0.4]
        alphas, betas = [2, 0.9], [0.1, 5]

        gammars = bernoulli.rvs(pis[0], size=n_samples)
        n_pos_events = sum(gammars)
        n_neg_events = n_samples - n_pos_events

        ys_of_type1 = betabinom.rvs(n_trials, alphas[0], betas[0], size=n_pos_events)
        ys_of_type2 = betabinom.rvs(n_trials, alphas[1], betas[1], size=n_neg_events)

        ys = np.concatenate((ys_of_type1, ys_of_type2))
        ns = np.ones(n_samples, dtype=np.int) * n_trials

        em_mbb = MixtureBetaBinomial(
            n_components=2,
            max_m_step_iter=250,
            tor=1e-6)

        params = em_mbb.fit((ys, ns), max_iters=200, init_methods="mixbin")
        print(params)
        print(alphas, betas, pis)

    """

    def __init__(self,
                 n_components=2,
                 max_m_step_iter=250,
                 tor=1e-6,
                 n_init_searches=10
                 ):
        """Initialization method

        Args:
            n_components (int): number of mixtures. Defaults to 2.
            max_m_step_iter (int): maximum number iterations for the sub-optimiaztion 
                routine at M-step. Defaults to 250.
            tor ([type], optional): [description]. Defaults to 1e-6.
            n_init_searches (int): number of trials for good param initial values
        """
        super(MixtureBetaBinomial, self).__init__()
        self.n_components = n_components
        self.max_m_step_iter = max_m_step_iter
        self.tor = tor
        self.n_init_searches = n_init_searches

    def log_likelihood_betabin(self, y, n, a, b, pi=1.0):
        """log likelihood of data under a component of beta-binomial distribution

        Args:
            y (np.array): number of positive events
            n (np.array): total number of trials
            a (float): alpha
            b (float): beta
            pi (float): weight of mixture component.

        Returns:
            np.array: log_likelihood of data
        """
        return gammaln(y + a) + gammaln(n - y + b) + gammaln(a + b) - \
               (gammaln(a) + gammaln(b) + gammaln(n + a + b)) + np.log(pi)

    def log_likelihood_mixbetabin(self, y, n, params):
        """log likelihood of the dataset under mixture beta-binomial distribution

        Args:
            y (np.array): number of positive events
            n (np.array): total number of trials
            params (np.array): parameters of the model

        Returns:
            float: log likelihood of the dataset
        """
        logLik_mat = np.zeros((len(y), self.n_components), dtype=float)
        for k in range(self.n_components):
            a, b, pi = params[k], params[k +
                                         self.n_components], params[k + 2 * self.n_components]
            logLik_mat[:, k] = self.log_likelihood_betabin(y, n, a, b, pi)
        return logsumexp(logLik_mat, axis=1).sum()

    def E_step(self, y, n, params):
        """Expectation step

        Args:
            y (np.array): number of positive events
            n (np.array): total number of trials
            params (np.array): parameters of the model

        Returns:
            np.array: expectation of the latent variables
        """
        log_E_gammas = [None] * self.n_components
        for k in range(self.n_components):
            a, b, pi = params[k], params[k +
                                         self.n_components], params[k + 2 * self.n_components]
            log_E_gammas[k] = np.log(pi) + gammaln(y + a) + gammaln(n - y + b) + gammaln(a + b) - \
                              (gammaln(a) + gammaln(b) + gammaln(n + a + b))

        # normalize as they haven't been
        log_E_gammas = log_E_gammas - logsumexp(log_E_gammas, axis=0)
        return np.exp(log_E_gammas)

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
        params = self._update_pis(E_gammas, params, len(n))
        bounds = self._get_bounds()
        nll_fun = self._get_nnl_fun()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = np.random.get_state()
            np.random.seed(42)
            res = minimize(nll_fun, x0=params[:2 * self.n_components],
                           args=(y, n, E_gammas, self.n_components,
                                 params[2 * self.n_components:3 * self.n_components]),
                           bounds=bounds,
                           method='L-BFGS-B',  # 'SLSQP',
                           options={'disp': False,
                                    'maxiter': self.max_m_step_iter},
                           )
            np.random.set_state(state)
        params[:2 * self.n_components] = res.x
        return params

    def _update_pis(self, E_gammas, params, N_samples):
        """analytics updates for pis:
            $\pi_k = \frac{\sum_{i=1}^N \hat{\gamma}_K^{i}}
            {\sum_{i=1}^N \sum_{k=1}^K \hat{\gamma}_K^{i}}$

        Returns:
            list of floats: updated params
        """
        for k in range(self.n_components):
            E_gammas[k][E_gammas[k] == 0] = 1e-20
            params[2 * self.n_components + k] = np.sum(E_gammas[k]) / N_samples
        return params

    def _get_bounds(self):
        """generate none-negative bounds for model parameters

        Returns:
            list of tuples: non-negative bounds in the input format of Scipy minimize function 
        """
        bounds = [(0, None)] * (2 * self.n_components)
        return tuple(bounds)

    def _get_nnl_fun(self):
        """get the loss function required by Scipy minimize function
        """

        def neg_log_likelihood(params, *args):
            """negative log likelihood for M-step optimization

            Args:
                params (array): model parameters: [a_0, a_1, a_2, ..., b_0, b_1, b_2, ..., pi_0, pi_1, pi_2,...]
                args: list of function arguments, 
                    args[0] (np.array): y
                    args[1] (np.array): n 
                    args[2] (np.array): is \hat{\gamma}, expectation of the latent variables, E-step results
                    args[3] (int): is the number of components
            Returns:
                [type]: [description]
            """
            y, n = args[0], args[1]
            E_gammas = args[2]
            n_components = args[3]
            pis = args[4]

            log_likelihood = 0
            for k in range(n_components):
                a, b, pi = params[k], params[k +
                                             n_components], pis[k]
                E_gamma_k = E_gammas[k]
                log_pdf = E_gamma_k * (
                        np.log(pi) + gammaln(n + 1) + gammaln(y + a) + gammaln(n - y + b) + gammaln(a + b) -
                        (gammaln(y + 1) + gammaln(n - y + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b)))
                log_likelihood += np.sum(log_pdf)
            return -log_likelihood

        return neg_log_likelihood

    def _perturb(self, array):
        """Add noise to perturb the data

        Args:
            array (np.array): array of data under perturbation

        Returns:
            np.array: data with noise 
        """
        array += np.random.uniform(-0.025, 0.025, np.shape(array))
        while np.any(array) < 0 or np.any(array) > 1:
            array += np.random.uniform(-0.025, 0.025, np.shape(array))
        return array

    def _init_with_kmeans(self, y, n):
        """Warm initialization with K-means

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials

        Returns:
            np.array: gammas
            np.array: randomly initialized parameters of the model
        """
        sample_size = len(y)
        proportions = y / n
        kmeans = KMeans(
            n_clusters=self.n_components,
            max_iter=500,
            random_state=44
        )
        kmeans.fit(proportions.reshape(-1, 1))
        gammars = np.zeros((sample_size, self.n_components), dtype=float)
        # deterministic gammars, to be perturbed
        gammars[range(sample_size), kmeans.labels_] = 1.0

        pi = np.sum(gammars, axis=0)
        pi = pi / np.sum(pi)

        params = np.concatenate([np.random.uniform(0.6, 0.9, self.n_components),
                                 np.random.uniform(
                                     0.6, 0.9, self.n_components),
                                 pi])

        return self._perturb(gammars.T), params

    def _init_with_mixbin(self, y, n):
        """Initialization with mixture of Binomials

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials

        Returns:
            np.array: gammas
            np.array: randomly initialized parameters of the model
        """
        em_mb = MixtureBinomial(
            n_components=self.n_components,
            tor=1e-6,
        )
        mb_params = em_mb.fit((y, n), max_iters=100, early_stop=True)
        gammars = em_mb.E_step(y, n, mb_params)

        # alpha + beta = concentration
        # alpha/(alpha + beta) = p
        ps = mb_params[:self.n_components]
        concentrations = np.random.uniform(9.0, 11.0, self.n_components)
        init_alphas = ps * concentrations
        init_betas = (1 - ps) * concentrations

        params = np.concatenate([init_alphas,
                                 init_betas,
                                 mb_params[self.n_components:]
                                 ])
        return gammars, params

    def _param_init(self, y, n, init_method="mixbin"):
        """Initialziation of model parameters

        Currently implemented methods:
            1. random: random initialization
            2. kmeans: K-means
            3. mixbin: mixture of Binomial

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            init_method (string): one of "kmeans", "mixbin", "random"

        Returns:
            np.array: initialized model parameters
        """
        if init_method == "random":
            params = np.concatenate([
                np.random.uniform(0.6, 0.9, 2 * self.n_components),
                np.random.uniform(0.4, 0.6, self.n_components)
            ])
            return params
        if init_method == "kmeans":
            E_gammas, params = self._init_with_kmeans(y, n)
            return self.M_step(y, n, E_gammas, params)
        if init_method == "mixbin":
            E_gammas, params = self._init_with_mixbin(y, n)
            return self.M_step(y, n, E_gammas, params)
        # for inner test
        if init_method == "fixed":
            params = np.concatenate([
                np.ones(2 * self.n_components, dtype=np.float),
                np.ones(self.n_components, dtype=np.float) * 0.6
            ])
            return params

        raise Exception(
            'Invalid initialization method {}, please specify one of "kmeans", "mixbin", "random"'.format(init_method))

    def EM(self, y, n, params, max_iters=250, early_stop=True,
           verbose=False, n_tolerance=10):
        """EM algorithim

        Args:
            y (np.array): number of positive events
            n (np.array): total number of trials respectively
            max_iters (int): maximum number of iterations for EM. Defaults to 250.
            params (np.array): init params.
            early_stop (bool): whether early stop training. Defaults to False.
            verbose (bool): whether print training information. Defaults to False.
            n_tolerance (int): the max number of violations to trigger early stop.

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

            # current NLL loss
            losses.append(-self.log_likelihood_mixbetabin(y, n, params))

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

    def fit(self, data, max_iters=250, init_method="mixbin", early_stop=False,
            pseudocount=0.1, n_tolerance=5, verbose=False):
        """model training.

        Args:
            data (tuple of arrays): y, n: number of positive events and total number of trials respectively
            max_iters (int): maximum number of iterations for EM. Defaults to 250.
            init_method (string): one of the initialization methods: "kmeans", "mixbin", or "random"
            early_stop (bool): whether early stop training. Defaults to False.
            pseudocount (float) : add pseudocount if data is zero
            n_tolerance (int): the max number of violations to trigger early stop.
            verbose (bool): whether print training information. Defaults to False.

        Returns
            np.array: trained parameters
        """
        y, n = self._preprocess(data, pseudocount)

        best_param, losses = None, [sys.maxsize]
        for _ in range(self.n_init_searches):
            init_params = self._param_init(y, n, init_method)
            self.EM(y, n, init_params, max_iters=500, early_stop=True)
            if self.losses[-1] < losses[-1]:
                best_param, losses = init_params, self.losses
        if verbose:
            print("=" * 25)
            print("Init params: {}".format(best_param))
            print("=" * 25)
        params = self.EM(y, n, best_param, max_iters=max_iters,
                         early_stop=early_stop, verbose=verbose,
                         n_tolerance=n_tolerance)
        self.losses = losses + self.losses  # because we continue EM on the best init_params
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

        alphas = self.params[:self.n_components]
        betas = self.params[self.n_components: 2 * self.n_components]
        pis = self.params[2 * self.n_components: 3 * self.n_components]

        labels = np.random.choice(self.n_components, size=n_trials.shape, p=pis)
        ys_out = np.zeros(n_trials.shape, dtype=np.int32)
        for i in range(self.n_components):
            _idx = np.where(labels == i)
            ys_out[_idx] = betabinom.rvs(n_trials[_idx].astype(np.int32), alphas[i], betas[i])

        return ys_out


if __name__ == "__main__":
    n_samples = 2000
    n_trials = 1000
    pis = [0.6, 0.4]
    alphas, betas = [2, 0.9], [0.1, 5]

    gammars = bernoulli.rvs(pis[0], size=n_samples)
    n_pos_events = sum(gammars)
    n_neg_events = n_samples - n_pos_events

    ys_of_type1 = betabinom.rvs(
        n_trials, alphas[0], betas[0], size=n_pos_events)
    ys_of_type2 = betabinom.rvs(
        n_trials, alphas[1], betas[1], size=n_neg_events)

    ys = np.concatenate((ys_of_type1, ys_of_type2))
    ns = np.ones(n_samples, dtype=np.int) * n_trials

    em_mbb = MixtureBetaBinomial(
        n_components=2,
        max_m_step_iter=250,
        tor=1e-6)

    params = em_mbb.fit((ys, ns), max_iters=200,
                        init_method="random", early_stop=False)
    print(params)
    print(em_mbb.model_scores)
    params = em_mbb.fit((ys, ns), max_iters=200, init_method="kmeans")
    print(params)
    print(em_mbb.model_scores)
    params = em_mbb.fit((ys, ns), max_iters=200, init_method="mixbin")
    print(params)
    print(em_mbb.model_scores)
    print(alphas, betas, pis)
