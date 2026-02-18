import pandas as pd
import numpy as np
from scipy import special, optimize
from scipy.stats import t, kurtosis
from scipy.integrate import quad

# neg_T_loglikelihood(mu, s, nu, x) (edited from 7.2)
def neg_T_loglikelihood(mu, s, nu, x):
    x = np.asarray(x, dtype=float)
    n = x.size
    np12 = (nu + 1.0) / 2.0
    # loggamma((nu+1)/2) - loggamma(nu/2) - log(sqrt(pi*nu)*s)
    const = (special.gammaln(np12) - special.gammaln(nu / 2.0) - np.log(np.sqrt(np.pi * nu) * s))
    xm = 1.0 + ((x - mu) / s) ** 2 / nu
    ll = n * const - np12 * np.sum(np.log(xm))
    return -ll

def fit_general_t(x):
    x = np.asarray(x, dtype=float)
    # starting values
    start_m = float(np.mean(x))
    ex_kurt = float(kurtosis(x, fisher=True, bias=False)) # excess kurtosis
    ex_kurt = max(ex_kurt, 1e-6) # avoid divide-by-zero
    start_nu = float(6.0 / ex_kurt + 4.0)

    v = float(np.var(x, ddof=1))
    start_s = float(np.sqrt(v * (start_nu - 2.0) / start_nu)) if start_nu > 2 else float(np.sqrt(v))

    # constraints:
    # s >= 1e-6
    # nu >= 2.0001
    theta0 = np.array([
        start_m,
        np.log(max(start_s - 1e-6, 1e-8)),
        np.log(max(start_nu - 2.0001, 1e-8)),
    ], dtype=float)

    # s  = 1e-6   + exp(theta[1])
    # nu = 2.0001 + exp(theta[2])
    def objective(theta):
        mu = float(theta[0])
        s = 1e-6 + float(np.exp(theta[1]))
        nu = 2.0001 + float(np.exp(theta[2]))
        return neg_T_loglikelihood(mu, s, nu, x)
    
    # Nelder-Mead （from 7.2)
    res = optimize.minimize(
        objective,
        x0=theta0,
        method="Nelder-Mead",  # your style; teacher uses Ipopt, but same objective
        options={"maxiter": 200000, "xatol": 1e-14, "fatol": 1e-14},
    )

    mu_hat = float(res.x[0])
    sigma_hat = float(1e-6 + np.exp(res.x[1]))
    nu_hat = float(2.0001 + np.exp(res.x[2]))
    return mu_hat, sigma_hat, nu_hat

# The following functions are from Week05/RiskStats.jl
# VaR(a)
def VaR(a, alpha=0.05):
    x = np.sort(np.asarray(a, dtype=float))
    n = x.size

    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))

    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    return -v

# ES(a)
def ES(a, alpha=0.05):
    x = np.sort(np.asarray(a, dtype=float))
    n = x.size

    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))

    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    es = float(np.mean(x[x <= v]))
    return -es

# Read
cin = pd.read_csv("data/test7_2.csv").values
x = cin[:, 0].astype(float)

# fit general t
m, s, nu = fit_general_t(x)

# Simulate
u = np.random.rand(10000)
sim = t.ppf(u, df=nu, loc=m, scale=s)

out = pd.DataFrame({
    "ES Absolute": [ES(sim, alpha=0.05)],
    "ES Diff from Mean": [ES(sim - np.mean(sim), alpha=0.05)],
})
out.to_csv("testout8_6.csv", index=False, float_format="%.17g")