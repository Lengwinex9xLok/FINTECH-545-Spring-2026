import pandas as pd
import numpy as np
from scipy.stats import t

# This code is followed from Week05/RiskStats.jl/function VaR(d::UnivariateDistribution; alpha=0.05)
def VaR(d, alpha=0.05):
    return -d.ppf(alpha)

# read
cin = pd.read_csv("data/test7_2.csv").values
x = cin[:, 0]

# From test_setup.jl
# Julia: fd = fit_general_t(cin[:,1])
nu, m, s = t.fit(x)

errorModel = t(df=nu, loc=m, scale=s)

var_abs = VaR(errorModel)
var_diff = -t(df=nu, loc=0.0, scale=s).ppf(0.05)

out = pd.DataFrame({"VaR Absolute": [var_abs],"VaR Diff from Mean": [var_diff]})
out.to_csv("testout8_2.csv", index=False, float_format="%.17g")

