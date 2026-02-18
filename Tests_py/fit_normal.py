import numpy as np

def fit_normal(x):
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    return mu, sigma