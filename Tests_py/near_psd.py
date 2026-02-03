import numpy as np

# This function is based on week03.jl - Near PSD matrix (function near_psd(a; epsilon=0.0))
# Near PSD matrix
def near_psd(a, epsilon=0.0):
    out = np.array(a, dtype=float, copy=True)
    out = (out + out.T) / 2.0 # Julia: out = copy(a)

    diag = np.diag(out)
    is_corr = np.all(np.isclose(diag, 1.0, rtol=1e-8, atol=1e-12)) # Julia: count (x-> x ≈ 1.0, diag(out)) != n)

# If input is covariance, convert to correlation
    invSD = None
    if not is_corr:
        sd = np.sqrt(np.maximum(diag, 0.0)) # Standard deviations
        sd_safe = np.where(sd == 0.0, 1.0, sd) # ground against zero variance
        invSD = np.diag(1.0 / sd_safe)
        out = invSD @ out @ invSD
        out = (out + out.T) / 2.0 # symmetry

    vals, vecs = np.linalg.eigh(out) # Julia: vals, vecs = eigen(out)
    vals = np.maximum(vals, epsilon) # Julia: vals = max.(vals, epsilon)

# Julia: T = 1 ./ (vecs .* vecs * vals)
    vecs_sq = vecs * vecs
    denom = vecs_sq @ vals
    denom = np.where(denom == 0.0, 1.0, denom)

# Julia: T = diagm(sqrt.(T))
    T_vec = 1.0 / denom
    T = np.diag(np.sqrt(T_vec))

# Julia: L = diagm(sqrt.(vals))
    L = np.diag(np.sqrt(vals))

# Julia: B = T * vecs * L
# reconstruct PSD matrix
    B = T @ vecs @ L
    out = B @ B.T
    out = (out + out.T) / 2.0

# scale back if input is covariance
# Julia: invSD = diagm(1 ./ diag(invSD))
    if invSD is not None:
        sd = 1.0 / np.diag(invSD) # recovered SD
        SD = np.diag(sd)
        out = SD @ out @ SD
        out = (out + out.T) / 2.0

    return out