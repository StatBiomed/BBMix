# -*- coding: utf-8 -*-
"""binomial_mix.

Chen Qiao: cqiao@connect.hku.hk
"""

import numpy as np
from scipy.stats import entropy


def bic_criterion(k, n, nll):
    """Bayesian Information Criterion (BIC)

    Args:
        k (int): the number of estimated parameters
        n (int): the number of datapoints, i.e., sample size
        nll (float): negative log likelihood of the dataset given the model
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
    return np.sum(entropy(probs, axis=1))


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
