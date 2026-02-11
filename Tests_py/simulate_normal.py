import numpy as np
from chol_psd import chol_psd
from near_psd import near_psd

# This code is followed from week05/simulate.jl -simulateNormal
def simulate_normal(N: int, cov, mean=None, seed: int = 1234, fix_method=near_psd):
    # Julia: n, m = size(cov); if n != m throw...
    # For error check
    cov = np.array(cov, dtype=float, copy=True)
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    # Julia: _mean = fill(0.0,n); if !isempty(mean) copy!(_mean, mean)
    # For mean check
    if mean is None or (isinstance(mean, (list, tuple, np.ndarray)) and len(mean) == 0):
        _mean = np.zeros(n, dtype=float)
    else:
        _mean = np.array(mean, dtype=float).reshape(-1)
        if _mean.size != n:
            raise ValueError(f"Mean ({_mean.size}) is not the size of cov ({n},{n})")

    # Julia: take the root
    # Cholesky if PD; if not PD, chol_psd(PSD) and then fixMethod(cov)(near_psd)
    try:
        # Julia:l = Matrix(cholesky(cov).L)
        L = np.linalg.cholesky(0.5 * (cov + cov.T))
    except np.linalg.LinAlgError:
        # Julia: catch PosDefException:
        # try chol_psd!(l,cov) catch chol_psd!(l, fixMethod(cov))
        try:
            L = chol_psd(cov)
            # NaN case
            if np.isnan(L).any():
                raise ValueError("chol_psd produced NaN")
        except Exception:
            cov_fixed = fix_method(cov)
            L = chol_psd(cov_fixed)

    np.random.seed(seed)                      # Julia: Random.seed!(seed)
    out = np.random.randn(n, N)               # Julia: d=Normal(0.0,1.0) + rand!(d,out)

    # apply the standard normals to the cholesky root
    # Julia: out = (l*out)'
    out = (L @ out).T

    # Loop over itereations and add the mean
    # Julia: for i in 1:n
    #    out[:,i] = out[:,i] .+ _mean[i]
    out += _mean.reshape(1, n)

    # Return N×n
    return out

