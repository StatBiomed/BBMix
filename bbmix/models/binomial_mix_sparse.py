# -*- coding: utf-8 -*-
"""binomial_mix.

Chen Qiao: cqiao@connect.hku.hk
"""

import numpy as np

from scipy.special import gammaln, logsumexp
from .model_base import ModelBase
from .statistics import bic_criterion, entropy_criterion


class MixtureBinomialSparseBatch(ModelBase):
    """Mixture of Binomial Models fitted on Batched Sparse Matrices

    This class implements EM algorithm for parameter estimation of Mixture 
    Binomial models, simultaneously fitting batched samples. 

    Attributes:
        n_components (int): number of mixtures.
        tor (float): tolarance difference for earlier stop training.
        params (2-d float array): parameters of the model, [[p_1, p_2, ..., 
            p_K, pi_1, pi_2, ..., pi_K]],
            None before parameter estimation.
        losses (2-D array): Batch list of negative loglikelihood losses of the training 
            process, None before parameter estimation.
        model_scores (dict{name:2-d array}): scores for the model, including "BIC" and "ICL" scores

    Notes
    -----
    Because M-step has analytical solution, parameter estimation is fast.

    Usage:

        em_mb = MixtureBinomialSparseBatch(
        n_components=2,
        tor=1e-6)
        params = em_mb.fit((ys, ns), valid_sample_sizes, max_iters=250, early_stop=True)

    Simulation experiment:

        import numpy as np
        from scipy.stats import bernoulli, binom
        from bbmix.models import MixtureBinomialSparseBatch
        
        n_samples = [2000, 1000]
        n_trials = [1000, 500]
        # p1, p2, pi1, pi2
        param_batch = [[0.4, 0.8, 0.6, 0.4], [0.2, 0.7, 0.3, 0.7]]
        
        flat_ys, flat_ns = [], []
        for params, n_smp, n_tri in zip(param_batch, n_samples, n_trials):
            p1, p2, pi1, pi2 = params
            gammars = bernoulli.rvs(pi1, size=n_smp)
            n_pos_events = sum(gammars)
            n_neg_events = n_smp - n_pos_events
        
            ys_of_type1 = binom.rvs(n_tri, p1, size=n_pos_events)
            ys_of_type2 = binom.rvs(n_tri, p2, size=n_neg_events)
        
            ys = np.concatenate((ys_of_type1, ys_of_type2))
            ns = np.ones(n_smp, dtype=np.int) * n_tri
            flat_ys.append(ys)
            flat_ns.append(ns)
        
        flat_ys, flat_ns = np.concatenate(flat_ys), np.concatenate(flat_ns)
        valid_row_samples = n_samples
        
        em_mb = MixtureBinomialSparseBatch(
            n_components=2,
            tor=1e-20)
    
        params = em_mb.fit((flat_ys, flat_ns), 
                           valid_row_samples, 
                           max_iters=250, 
                           early_stop=True,
                           verbose=True)
        print(params)
        print(param_batch)
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
        super(MixtureBinomialSparseBatch, self).__init__()
        self.n_components = n_components
        self.tor = tor

    def E_step(self, y, n, repeated_params):
        """Expectation step

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            repeated_params (np.array): parameters of the model repeated for each sample in batch
        Returns:
            np.array: expectation of the latent variables
        """
        params = repeated_params
        E_gammas = np.empty((self.n_components, y.shape[0]))
        for k in range(self.n_components):
            p_k, pi_k = params[:, k], params[:, k + self.n_components]
            E_gammas[k] = y * np.log(p_k) + (n - y) * \
                          np.log(1 - p_k) + np.log(pi_k)

        # normalize as they havn't been
        E_gammas = E_gammas - logsumexp(E_gammas, axis=0, keepdims=True)
        return np.exp(E_gammas)

    def M_step(self, y, n, E_gammas, params, valid_row_sizes):
        """Maximization step

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            E_gammas (np.array): results of E step
            params (np.array): model parameters 
            valid_row_sizes (np.array): sample sizes in this batch

        Returns:
            np.array: updated model parameters
        """
        E_gammas[E_gammas == 0] = 1e-20 # for numerical issues
        for k in range(self.n_components):
            params[:, k] = (aggregate_by_batch(y * E_gammas[k], valid_row_sizes) /  
                                aggregate_by_batch(n * E_gammas[k], valid_row_sizes))
            params[:, k + self.n_components] = (aggregate_by_batch(E_gammas[k], valid_row_sizes) / 
                                                valid_row_sizes)
        return params

    def log_likelihood_binomial(self, y, n, p, pi=1.0):
        """log likelihood of data under binomial distribution

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            p (np.array): probabilities of positive event
            pi (np.array): weights of mixture component

        Returns:
            np.array: log likelihood of data
        """
        return gammaln(n + 1) - (gammaln(y + 1) + gammaln(n - y + 1)) \
               + y * np.log(p) + (n - y) * np.log(1 - p) + np.log(pi)

    def log_likelihood_mixture_bin(self, y, n, repeated_params, valid_row_sizes):
        """log likelihood of dataset under mixture of binomial distribution

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            repeated_params (np.array): parameters of the model repeated for each sample in batch
            valid_row_sizes (np.array): sample sizes in this batch
        Returns:
            1-d float array: log likelihoods of the batch samples
        """
        params = repeated_params
        logLik_mat = np.zeros((self.n_components, y.shape[0]), dtype=float)
        for k in range(self.n_components):
            p_k, pi_k = params[:, k], params[:, k + self.n_components]
            logLik_mat[k] = self.log_likelihood_binomial(y, n, p_k, pi_k)
        return aggregate_by_batch(logsumexp(logLik_mat.T, axis=-1), valid_row_sizes)

    def EM(self, y, n, params, valid_row_sizes, max_iters=250, early_stop=False, n_tolerance=10,
           loss_buffer=20,
           verbose=False):
        """EM algorithim

        Args:
            y (np.array): number of positive events
            n (np.array): total number of trials respectively
            params (list): init model params
            valid_row_sizes (array): sample sizes in this batch
            max_iters (int, optional): maximum number of iterations for EM. Defaults to 250.
            early_stop (bool, optional): whether early stop training. Defaults to False.
            n_tolerance (int): the max number of violations to trigger early stop.
            pseudocount (float) : add pseudocount if data is zero
            loss_buffer (int): maximum history of steps kept for losses
            verbose (bool, optional): whether print training information. Defaults to False.

        Returns:
            np.array: trained parameters
        """
        batch_size = np.shape(valid_row_sizes)[0]
        n_tol = n_tolerance
        losses = np.zeros((batch_size, loss_buffer))

        # 2-D array
        repeated_params = np.repeat(params, valid_row_sizes, axis=0)
        for ith in range(max_iters):

            # E step
            E_gammas = self.E_step(y, n, repeated_params)

            # M step
            tmp_params = self.M_step(y, n, E_gammas, params, valid_row_sizes)
            repeated_params = np.repeat(tmp_params, valid_row_sizes, axis=0)

            # record current NLL loss
            losses[:, ith % loss_buffer] = -self.log_likelihood_mixture_bin(y, n, 
                                                                            repeated_params, 
                                                                            valid_row_sizes)

            if verbose:
                print("=" * 10, "Iteration {}".format(ith + 1), "=" * 10)
                print("Max Negative LogLikelihood Loss: {}".format(
                    np.max(losses[:, ith % loss_buffer])))
                print("=" * 25)
            
            loss_diffs = losses[:, (ith-1)% loss_buffer] - losses[:, ith % loss_buffer]
            is_improving = loss_diffs >= self.tor
            
            # update repeated params
            params[is_improving] = tmp_params[is_improving]
            repeated_params = np.repeat(params, valid_row_sizes, axis=0)
            
            # max_improvement = np.max(loss_diffs)
            if early_stop:
                if np.sum(is_improving) == 0:
                    n_tol -= 1
                else:
                    n_tol = n_tolerance
                if n_tol == 0:
                    if verbose:
                        print("Improvement halts, early stop training.")
                    break

        self.score_model(np.shape(params)[1],
                         valid_row_sizes,
                         losses[:, ith % loss_buffer], 
                         E_gammas.T)
        self.params = params
        self.losses = np.hstack([losses[:, :ith % loss_buffer+1], losses[:, ith % loss_buffer+1:]])
        return params
    
    def score_model(self, k, valid_row_sizes, nll, probs):
        """score model using BIC criterion

        Args:
            k (int): the number of estimated parameters
            valid_row_sizes (int): the number of datapoints, i.e., sample size
            nll (float): negative log likelihood of the dataset given the model
            probs (np.array/np.ndarray): batch probabilistic distributions of discrete random variables (latent)

        Returns:
            dict: BIC and ICL scores of the given model on the dataset
        """
        bic_score = bic_criterion(k, valid_row_sizes, nll)
        entropy_score = aggregate_by_batch(
            probs, valid_row_sizes, 
            agg_fun=lambda probs: entropy_criterion(probs.T)
            )
        icl_score = bic_score + entropy_score
        self.model_scores = {"BIC": bic_score, "ICL": icl_score}
        return self.model_scores

    def _param_init(self, y, n, batch_size=None):
        """Initialziation of model parameters

        Args:
            y (np.array): number of positive events
            n (np.array): number of total trials
            batch_size (int): number of samples in this batch
        Returns:
            np.array (2-d): initialized model parameters
        """
        return np.concatenate([np.random.uniform(0.49, 0.51, (batch_size, self.n_components)),
                               np.random.uniform(0.4, 0.6, (batch_size, self.n_components))], axis=1)

    def fit(self, data, valid_row_sizes, max_iters=250, early_stop=False, pseudocount=0.1,
            n_tolerance=10, loss_buffer=20, verbose=False):
        """Fit function

        Args:
            data (tuple of arrays): y, n: number of positive events and total number of trials respectively
            valid_row_sizes (np.array): segment input batch input specific samples which have different sizes
            max_iters (int, optional): maximum number of iterations for EM. Defaults to 250.
            early_stop (bool, optional): whether early stop training. Defaults to False.
            pseudocount (float) : add pseudocount if data is zero
            n_tolerance (int): the max number of violations to trigger early stop.
            loss_buffer (int): maximum history of steps kept for losses
            verbose (bool, optional): whether print training information. Defaults to False.

        Returns:
            np.array: trained parameters
        """
        batch_size = np.shape(valid_row_sizes)[0]
        y, n = self._preprocess(data, pseudocount)
        
        init_params = self._param_init(y, n, batch_size)
        if verbose:
            for ith, param in enumerate(init_params):
                print("=" * 25)
                print("# {} Init params: {}".format(ith, param))
                print("=" * 25)
        params = self.EM(y, n, init_params, valid_row_sizes, max_iters=max_iters,
                         early_stop=early_stop, verbose=verbose, loss_buffer=loss_buffer,
                         n_tolerance=n_tolerance)
        return params

    def sample(self, n_trials):
        """Generate data from fitted parameters
        n_trails :
        Args:
            n_trails (array_like): total number of trials

        Returns:
            2-d np.array: batch ys generated from the fitted distribution
        """
        if hasattr(self, 'params') == False:
            raise Exception("Error: please fit the model or set params before sample()")

        batchs = []
        for params in self.params:
            mus = params[:self.n_components]
            pis = params[self.n_components: 2 * self.n_components]
    
            labels = np.random.choice(self.n_components, size=n_trials.shape, p=pis)
            ys_out = np.zeros(n_trials.shape, dtype=int)
            for i in range(self.n_components):
                _idx = np.where(labels == i)
                ys_out[_idx] = binom.rvs(n_trials[_idx].astype(np.int32), mus[i])
            batchs.append(ys_out)
        return np.array(batchs)
    
    
def aggregate_by_batch(values, valid_row_sizes, agg_fun=np.sum, res_array=None):
    """Aggregate data by batch based on sample sizes
        Args:
            values (array): flattened values 
            valid_row_sizes (np.array): segment input batch input specific samples which have different sizes
            agg_fun (function): aggregated by this function
            res_array (array): record aggregation results, return new array if nor specified
        Returns:
            np.array: values aggregated by batch
    """
    left, batch_size = 0, len(valid_row_sizes)
    batch_res = res_array if res_array else np.empty(batch_size)
    for ith, smp_sz in enumerate(valid_row_sizes):
        right = left + smp_sz
        batch_res[ith] = agg_fun(values[left:right])
        left = right
    return batch_res
    

if __name__ == "__main__":
    import numpy as np
    from scipy.stats import bernoulli, binom
    from bbmix.models import MixtureBinomialSparseBatch
    
    n_samples = [2000, 1000]
    n_trials = [1000, 500]
    # p1, p2, pi1, pi2
    param_batch = [[0.4, 0.8, 0.6, 0.4], [0.2, 0.7, 0.3, 0.7]]
    
    flat_ys, flat_ns = [], []
    for params, n_smp, n_tri in zip(param_batch, n_samples, n_trials):
        p1, p2, pi1, pi2 = params
        gammars = bernoulli.rvs(pi1, size=n_smp)
        n_pos_events = sum(gammars)
        n_neg_events = n_smp - n_pos_events
    
        ys_of_type1 = binom.rvs(n_tri, p1, size=n_pos_events)
        ys_of_type2 = binom.rvs(n_tri, p2, size=n_neg_events)
    
        ys = np.concatenate((ys_of_type1, ys_of_type2))
        ns = np.ones(n_smp, dtype=np.int) * n_tri
        flat_ys.append(ys)
        flat_ns.append(ns)
    
    flat_ys, flat_ns = np.concatenate(flat_ys), np.concatenate(flat_ns)
    valid_row_samples = n_samples
    
    em_mb = MixtureBinomialSparseBatch(
        n_components=2,
        tor=1e-20)

    params = em_mb.fit((flat_ys, flat_ns), 
                       valid_row_samples, 
                       max_iters=250, 
                       early_stop=True,
                       verbose=True)
    print(params)
    print(param_batch)
    print(em_mb.model_scores)
