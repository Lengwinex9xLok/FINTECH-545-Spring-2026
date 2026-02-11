import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

# This code/function is followed from Week05/RiskStats.jl/function VaR(a)
def VaR(a, alpha=0.05):
    x = np.sort(np.asarray(a, dtype=float)) # Julia: sort(a)
    n = len(x)
    nup = int(np.ceil(n * alpha)) - 1 # Julia: ceil(na)
    ndn = int(np.floor(n * alpha)) - 1 # Julia: floor(na)
    v = 0.5 * (x[nup] + x[ndn])
    return -v

# This function is followed from Week05/week05.jl/function general_t_ll(mu,s,nu,x)
def _negloglike(params, x):
    mu, s, nu = params
    if s <= 1e-6 or nu <= 2.0001: # set constraint in Julia form
        return np.inf
    return -np.sum(t.logpdf(x, df=nu, loc=mu, scale=s))

# This function is followed from Week05/week05.jl/function fit_general_t(x)
def fit_general_t(x):
    x = np.asarray(x, dtype=float)

    # Julia: start_m = mean(x)
    start_m = float(np.mean(x))

    # Julia: start_nu = 6.0/kurtosis(x) + 4
    m2 = np.mean((x - start_m) ** 2)
    m4 = np.mean((x - start_m) ** 4)
    excess_kurt = (m4 / (m2 ** 2)) - 3.0
    start_nu = float(6.0 / max(excess_kurt,1e-6) + 4.0)

    # Julia: start_s = sqrt(var(x)*(start_nu-2)/start_nu)
    v = float(np.var(x, ddof=1))
    start_s = float(np.sqrt(v * (start_nu - 2.0) / start_nu))

    # @variable(mle, m, start=start_m)
    # @variable(mle, s>=1e-6, start=1)
    # @variable(mle, nu>=2.0001, start=start_s)
    bounds = [(None, None),(1e-6, None),(2.0001, None)]

    # Optimization
    res = minimize(_negloglike, [start_m, start_s, start_nu], args=(x,), method="L-BFGS-B", bounds=bounds)

    mu, s, nu = res.x
    return float(mu), float(s), float(nu)

# read
cin = pd.read_csv("data/test7_2.csv").values
x = cin[:, 0]

m, s, nu = fit_general_t(x)

# From test_setup.jl
# Julia: sim = fd.eval(rand(10000))
rng = np.random.RandomState(1234)
u = rng.rand(10000)
sim = t.ppf(u, df=nu, loc=m, scale=s)

var_abs = VaR(sim)
var_diff = VaR(sim - np.mean(sim))

out = pd.DataFrame({"VaR Absolute": [var_abs],"VaR Diff from Mean": [var_diff]})
out.to_csv("testout8_3.csv", index=False, float_format="%.17g")

