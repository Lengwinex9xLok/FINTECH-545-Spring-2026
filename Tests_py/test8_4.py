import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

alpha = 0.05

# The following functions are from Week05/RiskStats.jl
# VaR(d: UnivariateDistribution)
def VaR_dist(d, alpha=0.05):
    return -d.ppf(alpha)

# ES(d::UnivariateDistribution)
def ES_dist(d, alpha=0.05):
    v = VaR_dist(d, alpha)
    st = d.ppf(1e-12)
    integrand = lambda x: x * d.pdf(x)
    integral, _ = quad(integrand, st, -v, limit=200)
    return -integral / alpha

# Read
cin = pd.read_csv("data/test7_1.csv").values
x = cin[:, 0]

m = float(np.mean(x))
s = float(np.std(x, ddof=1))

# error model
d_abs = norm(loc=m, scale=s)

# Julia: ES(fd.errorModel)
es_abs = ES_dist(d_abs, alpha)

# Julia: ES(Normal(0, fd.errorModel.σ))
d_diff = norm(loc=0.0, scale=s)
es_diff = ES_dist(d_diff, alpha)

out = pd.DataFrame({
    "ES Absolute": [es_abs],
    "ES Diff from Mean": [es_diff]
})
out.to_csv("testout8_4.csv", index=False, float_format="%.17g")
