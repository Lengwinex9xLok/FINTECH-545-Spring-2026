import pandas as pd
import numpy as np

df = pd.read_csv("data/test1.csv")
X = df.to_numpy(dtype=float)

n, p = X.shape
# Build covariance matrix
cov = np.zeros((p, p), dtype=np.float64)

for i in range(p):
    for j in range(p):
        xi = X[:, i]
        xj = X[:, j]

        mask = ~np.isnan(xi) & ~np.isnan(xj) # Pairwise matching rows
        xi = xi[mask]
        xj = xj[mask]

        xi_mean = xi.mean() # sample covariance (n-1) (overlap)
        xj_mean = xj.mean()

        cov[i, j] = np.sum((xi - xi_mean) * (xj - xj_mean)) / (len(xi) - 1)

out = pd.DataFrame(cov, columns=df.columns)

out.to_csv("testout_1.3.csv", index=False, float_format="%.16g")


