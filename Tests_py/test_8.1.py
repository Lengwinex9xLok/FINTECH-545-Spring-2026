import pandas as pd
import numpy as np
from statistics import NormalDist

# This code is followed from Week05/RiskStats.jl/function VaR(d::UnivariateDistribution; alpha=0.05)
def VaR_dist(d, alpha=0.05):
    return -d.inv_cdf(alpha)

# read
cin = pd.read_csv("data/test7_1.csv").values
x = cin[:, 0]

# From test_setup.jl
# Julia: fd = fit_normal(cin[:,1])
m = float(np.mean(x))
s = float(np.std(x, ddof=1)) # default ddof=1

errorModel = NormalDist(m, s)

var_abs = VaR_dist(errorModel, 0.05)
var_diff = -NormalDist(0.0, s).inv_cdf(0.05)

out = pd.DataFrame({"VaR Absolute": [var_abs],"VaR Diff from Mean": [var_diff]})
out.to_csv("testout8_1.csv", index=False, float_format="%.17g")



