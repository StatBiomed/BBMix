# -*- coding: utf-8 -*-
"""binomial_mix.

Chen Qiao: cqiao@connect.hku.hk
"""

import numpy as np

from .statistics import bic_criterion, entropy_criterion


class ModelBase:
    """Base class of all the models

    This class encapsulated some common functions shared by all the derived models
    """

    def __init__(self):
        """Initialization
        """
        self.params = None
        self.losses = None
        self.model_scores = None

    def score_model(self, k, n, nll, probs):
        """score model using BIC and ICL criterions

        Args:
            k (int): the number of estimated parameters
            n (int): the number of datapoints, i.e., sample size
            nll (float): negative log likelihood of the dataset given the model
            probs (np.array/np.ndarray): batch probabilistic distributions of discrete random variables (latent)

        Returns:
            dict: BIC and ICL scores of the given model on the dataset
        """
        bic_score = bic_criterion(k, n, nll)
        entropy_score = entropy_criterion(probs)
        icl_score = bic_score + entropy_score
        self.model_scores = {"BIC": bic_score, "ICL": icl_score}
        return self.model_scores

    def _preprocess(self, data, pseudocount=0.1, min_n=10):
        """Preprocess inputs

        Args:
            data (tuple of arrays): y, n: number of positive events and total number of trials respectively
            pseudocount (float) : add pseudocount if data is zero
            min_n (int): minimum number of samples for filtering

        Returns
            tuple of np.array
        """
        y, n = data
        # y, n = y[n > min_n], n[n > min_n]  # filter extremes
        if np.any(y == 0):
            y = y.astype(float)
            y[y == 0] = pseudocount
        return y, n
