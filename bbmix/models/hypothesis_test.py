import numpy as np
from scipy.stats import chi2

def LR_test(LR, df=1, is_log=False):
    """Likelihood ratio test

    Args:
        LR (np.array): likelihood ratio at log scale between alternative model 
            vs null model, namely logLik difference
        df (int): degree of freedom in chi square distribution, namely number
            of additional parameters in alternative model
        is_log (bool): if return p value at log scale

    Returns:
        np.array: p value or log(p value) for single-sided test
    """
    if is_log:
        return chi2.logsf(2 * LR, df)
    else:
        return chi2.sf(2 * LR, df)