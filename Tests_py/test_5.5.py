import pandas as pd
import numpy as np
from simulate_pca import simulate_pca

cin = pd.read_csv("data/test5_2.csv").values

X = simulate_pca(cin, 100000, pctExp=0.99)

cout = np.cov(X, rowvar=False, ddof=1)

n = cin.shape[0]

pd.DataFrame(cout, columns=[f"x{i}" for i in range(1, n+1)]) \
    .to_csv("testout_5.5.csv", index=False, float_format="%.17g")
