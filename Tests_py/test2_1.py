import pandas as pd
import numpy as np

from Expo_Weighted_Cov import expo_weighted_cov

x = pd.read_csv("data/test2.csv")
X = x.to_numpy(dtype=np.float64)

cout = expo_weighted_cov(X, 0.97)

pd.DataFrame(cout, columns=x.columns).to_csv("testout_2.1.csv", index=False, float_format="%.16g")
