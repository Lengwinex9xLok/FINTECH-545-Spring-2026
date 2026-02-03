import pandas as pd
import numpy as np

df = pd.read_csv("data/test1.csv")
X = df.to_numpy(dtype=float)

n, p = X.shape
# Build covariance matrix
corr = np.zeros((p, p), dtype=np.float64)

for i in range(p):
    for j in range(i, p):
        xi = X[:, i]
        xj = X[:, j]

        mask = ~np.isnan(xi) & ~np.isnan(xj) # Pairwise matching rows
        xi_ = xi[mask]
        xj_ = xj[mask]


        if len(xi_) < 2:
            val = np.nan
        else: # This attempt is to generate 0.99999... from 1.0
            if i == j:
                s2 = np.cov(xi_, xi_, ddof=1)[0, 1]
                s = np.sqrt(s2)
                val = s2 / (s * s)
            else:
                val = np.corrcoef(xi_, xj_)[0, 1]

        corr[i, j] = val
        corr[j, i] = val

# Diagonal should be 1.0 (doesn't applied to this question)
# np.fill_diagonal(corr, 1.0) 

out = pd.DataFrame(corr, columns=df.columns)
# if don't mind diagonal to be 1.0 or 1
# out.to_csv("testout_1.4_1.csv", index=False, float_format="%.16g")

# Attempt to try to convert 1.0 to 0.99999...
# But failed...
out_str = out.astype(str)
out_str.to_csv("testout_1.4.csv", index=False, float_format="%.16g")
