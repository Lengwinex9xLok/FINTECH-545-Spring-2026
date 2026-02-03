import pandas as pd
import numpy as np

from Expo_Weighted_Cov import expo_weighted_cov

x = pd.read_csv("data/test2.csv")
X = x.to_numpy(dtype=np.float64)

cout_97 = expo_weighted_cov(X, 0.97)
sdev_97 = np.sqrt(np.diag(cout_97))

cout_94 = expo_weighted_cov(X, 0.94)
sdev_94 = 1.0 / np.sqrt(np.diag(cout_94))

D_97 = np.diag(sdev_97)
D_94 = np.diag(sdev_94)
cout = D_97 @ D_94 @ cout_94 @ D_94 @ D_97

pd.DataFrame(cout, columns=x.columns).to_csv("testout_2.3.csv", index=False, float_format="%.17g")

