import numpy as np
from scipy import special

def neg_T_loglikelihood(mu, s, nu, x):
    x = np.asarray(x, dtype=float)
    n = x.size
    np12 = (nu + 1.0) / 2.0
    # loggamma((nu+1)/2) - loggamma(nu/2) - log(sqrt(pi*nu)*s)
    const = (special.gammaln(np12) - special.gammaln(nu / 2.0) - np.log(np.sqrt(np.pi * nu) * s))
    xm = 1.0 + ((x - mu) / s) ** 2 / nu
    ll = n * const - np12 * np.sum(np.log(xm))
    return -ll