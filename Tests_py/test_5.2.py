import pandas as pd
import numpy as np
from simulate_normal import simulate_normal

cin = pd.read_csv("data/test5_2.csv").values

X = simulate_normal(100000, cin)

cout = np.cov(X, rowvar=False, ddof=1)

n = cin.shape[0]

pd.DataFrame(cout, columns=[f"x{i}" for i in range(1,n+1)]) \
  .to_csv("testout_5.2.csv", index=False, float_format="%.17g")