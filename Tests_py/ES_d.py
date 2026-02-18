import numpy as np
from scipy.integrate import quad

def VaR_dist(d, alpha=0.05):
    return -d.ppf(alpha)

def ES_dist(d, alpha=0.05):
    v = VaR_dist(d, alpha)
    st = d.ppf(1e-12)
    integrand = lambda x: x * d.pdf(x)
    integral, _ = quad(integrand, st, -v, limit=200)
    return -integral / alpha