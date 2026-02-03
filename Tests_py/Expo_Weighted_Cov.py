import numpy as np

# from w_{t-i} = (1-\lamda) \lambda^{i-1}
# Exponentially Weighted Covariance (Riskmetrics)
def expo_weighted_cov(X, lambdA):
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    ages = np.arange(n - 1, -1, -1) # from age = n-1 to age = 0
    w = (1.0 - lambdA) * (lambdA ** ages) # w_{t-i}
    w = w / w.sum()

    mu = (w[:, None] * X).sum(axis=0) # \mu = \sum w_t x_t
    Xm = X - mu

    cov = (Xm * w[:, None]).T @ Xm # cov = \sum w_t (X_t - \mu)(X_t - \mu)'
    return cov