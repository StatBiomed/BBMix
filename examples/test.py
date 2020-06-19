import bbmix
import numpy as np
from scipy import sparse
from scipy.io import mmread
import matplotlib.pyplot as plt

AD = mmread("../data/mitoDNA/cellSNP.tag.AD.mtx").tocsc().toarray()
DP = mmread("../data/mitoDNA/cellSNP.tag.DP.mtx").tocsc().toarray()
AD.shape, DP.shape

from bbmix.models import MixtureBetaBinomial

i = 1
idx = DP[i, :] >= 0
a = AD[i, idx]
d = DP[i, idx]
a, d

a = np.array([11.00000, 19.00000, 121.00000, 84.00000, 85.00000, 27.00000,
        5.00000, 78.00000, 341.00000, 52.00000, 91.00000, 19.00000,
        71.00000, 14.00000, 12.00000, 6.00000, 41.00000, 16.00000,
        385.00000, 20.00000, 26.00000, 12.00000, 29.00000, 135.00000,
        143.00000, 59.00000, 158.00000, 16.00000, 10.00000, 24.00000,
        21.00000, 29.00000, 33.00000, 12.00000, 33.00000, 626.00000,
        197.00000, 28.00000, 36.00000, 37.00000, 55.00000, 742.00000,
        44.00000, 21.00000, 21.00000, 14.00000, 61.00000, 134.00000,
        67.00000, 27.00000, 60.00000, 9.00000, 34.00000])
d = np.array([5952.00000, 7445.00000, 31233.00000, 21191.00000, 20790.00000,
        8253.00000, 3030.00000, 28684.00000, 99992.00000, 18046.00000,
        24853.00000, 5748.00000, 22853.00000, 8726.00000, 9948.00000,
        3217.00000, 11985.00000, 3223.00000, 119807.00000, 5305.00000,
        11373.00000, 4109.00000, 11661.00000, 37828.00000, 35186.00000,
        18033.00000, 48830.00000, 8656.00000, 6424.00000, 5464.00000,
        5573.00000, 7879.00000, 11385.00000, 4667.00000, 12877.00000,
        193484.00000, 59529.00000, 8497.00000, 13073.00000, 10614.00000,
        15670.00000, 238044.00000, 13759.00000, 4621.00000, 6112.00000,
        7365.00000, 21387.00000, 40508.00000, 17886.00000, 6968.00000,
        20506.00000, 3417.00000, 10248.00000])


model1 = MixtureBetaBinomial(n_components = 1, max_m_step_iter=500, tor=1e-20)
model2 = MixtureBetaBinomial(n_components = 2, max_m_step_iter=500, tor=1e-20)

for i in range(AD.shape[0]):
    a = AD[i, :]
    d = DP[i, :]
    idx = d >= 0
#     a, d = a[idx] + 1e-14, (d[idx] + 2e-14)
   
    print("======== {} =======".format(i))
    
    params1 = model1.EM((a, d), max_iters=500, init_method="mixbin", early_stop=False)
    print("-----------------")
    params2 = model2.EM((a, d), max_iters=500, init_method="mixbin", early_stop=True)
    p_val = bbmix.models.LR_test(model1.losses[-1] - model2.losses[-1], df = 3)
    print(i, "mode1: %.2f\tmodel2:%.2f\tp: %.3e" %(model1.losses[-1], model2.losses[-1], p_val))
    
    
from bbmix.models import MixtureBetaBinomial

model1 = MixtureBetaBinomial(n_components = 1, max_m_step_iter=500, tor=1e-20)
model2 = MixtureBetaBinomial(n_components = 2, max_m_step_iter=500, tor=1e-20)
