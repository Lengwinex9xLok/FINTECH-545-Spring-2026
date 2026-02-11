import numpy as np

# This code is followed from week05/simulate.jl -simulate_pca
def simulate_pca(a, nsim, pctExp=1.0, mean=None, seed=1234):
    a = np.array(a, dtype=float)
    n = a.shape[0]

    # Julia _mean = fill(0.0,n) then copy!(...)
    # If the mean is missing then set to 0, otherwise use provided mean
    if mean is None or (isinstance(mean, (list, tuple, np.ndarray)) and len(mean) == 0):
        _mean = np.zeros(n, dtype=float)
    else:
        _mean = np.array(mean, dtype=float).reshape(-1)

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)

    # julia returns values lowest to highest, flip them and the vectors
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    tv = np.sum(vals)

    # keep positive eigenvalues (>= 1e-8)
    pos = vals >= 1e-8
    vals = vals[pos]
    vecs = vecs[:, pos]

    if pctExp < 1.0:
        cumulative = np.cumsum(vals) / tv
        k = np.searchsorted(cumulative, pctExp) + 1
        vals = vals[:k]
        vecs = vecs[:, :k]

    # Julia: B = vecs * diag(sqrt(vals))
    B = vecs @ np.diag(np.sqrt(vals))

    # Julia: Random.seed!(seed)
    # m = size(vals,1)
    # r = randn(m, nsim)
    np.random.seed(seed)
    m = len(vals)
    r = np.random.randn(m, nsim)

    # Julia: out = (B*r)' 
    out = (B @ r).T

    out += _mean.reshape(1, n)

    return out
