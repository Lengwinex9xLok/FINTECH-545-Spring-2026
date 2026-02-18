import numpy as np

def VaR(a, alpha=0.05):
    x = np.sort(np.asarray(a, dtype=float))
    n = x.size

    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))

    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    return -v