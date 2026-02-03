import numpy as np

def higham_near_psd(pc, epsilon=1e-9, maxIter=100, tol=1e-9):
    pc = np.array(pc, dtype=float, copy=True)
    pc = (pc + pc.T) / 2.0

    diag = np.diag(pc)
    is_corr = np.all(np.isclose(diag, 1.0, rtol=1e-8, atol=1e-12))

    if not is_corr:
        sd = np.sqrt(np.maximum(diag, 0.0))
        sd_safe = np.where(sd == 0.0, 1.0, sd)

        corr = pc / (sd_safe[:, None] * sd_safe[None, :])
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)

        corr_psd = higham_near_psd(corr, epsilon=epsilon, maxIter=maxIter, tol=tol)

        # Scale back to covariance
        out = corr_psd * (sd_safe[:, None] * sd_safe[None, :])
        out = (out + out.T) / 2.0

        np.fill_diagonal(out, diag)

        return out
    
    deltaS = np.zeros_like(pc)
    Yk = pc.copy()
    norml = np.finfo(float).max
    i = 1

    while i <= maxIter:
        # Rk = Yk - deltaS
        Rk = Yk - deltaS

        # Projection onto PSD cone (Ps)
        eigvals, eigvecs = np.linalg.eigh(Rk)
        eigvals = np.maximum(eigvals, 0.0)
        Xk = (eigvecs * eigvals) @ eigvecs.T
        Xk = (Xk + Xk.T) / 2.0

        # Dykstra correction
        deltaS = Xk - Rk

        # Projection onto unit diagonal (Pu)
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1.0)
        Yk = (Yk + Yk.T) / 2.0

        # Convergence checks
        norm = np.linalg.norm(Yk - pc, ord="fro")
        minEigVal = np.min(np.linalg.eigvalsh(Yk))

        if (norm - norml) < tol and minEigVal > -epsilon:
            break

        norml = norm
        i += 1

    Yk = (Yk + Yk.T) / 2.0
    np.fill_diagonal(Yk, 1.0)

    return Yk