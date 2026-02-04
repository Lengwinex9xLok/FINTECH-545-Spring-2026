import numpy as np

# This function is based on week03.jl: Cholesky that assumes PD matrix
def chol_psd(a):
    a = np.array(a, dtype=float, copy=True) # Julia: n = size(a, 1)
    n = a.shape[0]

    root = np.zeros((n, n), dtype=float) # init root, Julia: root .= 0.0

    for j in range(n):
        if j > 0:
            s = root[j, :j] @ root[j, :j] # Julia: s = root[j, 1:(j-1)]' * root[j, 1:(j-1)]
        else:
            s = 0.0
        
        temp = a[j, j] - s # Similar to Julia: temp = a[j,j] .- s
        if temp <= 0.0 and temp >= -1e-8: # set the range to convert 0.0
            temp = 0.0
        
        root[j, j] = np.sqrt(temp) if temp >= 0.0 else np.nan # set a non-negative boundary in case sqrt

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                if j > 0:
                    s = root[i, :j] @ root[j, :j]
                else:
                    s = 0.0
                root[i, j] = (a[i, j] - s) * ir

    return root