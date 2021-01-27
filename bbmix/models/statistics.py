# -*- coding: utf-8 -*-
"""binomial_mix.

Chen Qiao: cqiao@connect.hku.hk
"""

import numpy as np
from scipy.stats import entropy
from scipy.special import gammaln, digamma


def bic_criterion(k, n, nll):
    """Bayesian Information Criterion (BIC)

    Args:
        k (int/np.array): the number of estimated parameters
        n (int/np.array): the number of datapoints, i.e., sample size
        nll (float/np.array): negative log likelihood of the dataset given the model
    returns:
        float: BIC score
    """
    return k*np.log(n) + 2*nll


def entropy_criterion(probs):
    """Entorpy of discrete probabilistic distributions

    Args:
        probs (np.array/np.ndarray): batch probabilistic distributions of discrete random variables 

    Returns:
        float: sum of entropy of all the random varaibles
    """
    return np.sum(entropy(probs, axis=-1), axis=-1)


def icl_criterion(k, n, nll, probs):
    """Integrated Completed Likelihood criterion (ICL)

    Args:
        k (int): the number of estimated parameters
        n (int): the number of datapoints, i.e., sample size
        nll (float): negative log likelihood of the dataset given the model
        probs (np.array/np.ndarray): batch probabilistic distributions of discrete random variables (latent)

    Returns:
        float: ICL score
    """
    return bic_criterion(k, n, nll) + entropy_criterion(probs)


def dirichlet_entropy(X, X_prior=None, axis=-1):
    """
    Get the entropy for dirichlet distributions. If X_prior is not None, 
    return the Kullback-Leibler divergence
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    
    Example
    -------
    theta_shapes1 = np.array([[0.3, 29.7], [3, 3], [29.7, 0.3]])
    theta_shapes2 = np.array([[364, 24197], [5886, 7475], [6075, 397]])
    dirichlet_entropy(theta_shapes2)
    dirichlet_entropy(theta_shapes2, theta_shapes1)
    """
    _entropy = (
        gammaln(X).sum(axis) -
        gammaln(X.sum(axis)) -
        ((X - 1) * digamma(X)).sum(axis) +
        ((X.sum(axis) - X.shape[axis]) * digamma(X.sum(axis)))
    )

    if X_prior is not None:
        _cross_entropy = (
            gammaln(X_prior).sum(axis) -
            gammaln(X_prior.sum(axis)) -
            ((X_prior - 1) * digamma(X)).sum(axis) +
            ((X_prior.sum(axis) - X_prior.shape[axis]) * digamma(X.sum(axis)))
        )
        _kl_divergence = _cross_entropy - _entropy

        return _kl_divergence
    else:
        return _entropy


def logfact(n):
    """
    Ramanujan's approximation of log n!
    https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
    """
    n = np.atleast_1d(n)
    n[n == 0] = 1  # for computational convenience
    return (
        n * (np.log(n) - 1)
        + np.log((n * (1. + 4. * (n * (1. + 2. * n))))) / 6.
        + np.log(np.pi) / 2.
    )


def logbincoeff(n, k):
    """
    Ramanujan's approximation of log [n! / (k! (n-k)!)]
    """
    n_logfact = logfact(n)
    k_logfact = logfact(k)
    n_sub_k_logfact = logfact(n - k)
    return n_logfact - k_logfact - n_sub_k_logfact


def normalize(X, axis=-1):
    """
    Normalization of tensor with sum to 1.
    
    Example
    -------
    X = np.random.rand(3, 5, 8)
    tensor_normalize(X, axis=1)
    """
    X_sum = np.sum(X, axis=axis, keepdims=True)
    return X / X_sum


def normalize_exp(X, axis=-1):
    """
    Amplify the log likelihood matrix by subtract the maximum.
    
    Example
    -------
    X = np.random.rand(3, 5, 8)
    normalize_exp(X, axis=1)
    """
    X_max = np.max(X, axis=axis, keepdims=True)
    return normalize(np.exp(X - X_max), axis=axis)