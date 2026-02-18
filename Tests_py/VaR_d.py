def VaR_dist(d, alpha=0.05):
    return -d.ppf(alpha)