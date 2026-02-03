import pandas as pd
import numpy as np

from Expo_Weighted_Cov import expo_weighted_cov

x = pd.read_csv("data/test2.csv")
X = x.to_numpy(dtype=np.float64)

cout = expo_weighted_cov(X, 0.94)

sdev = 1.0 / np.sqrt(np.diag(cout))
D = np.diag(sdev)

cout = D @ cout @ D

# if don't mind diagonal to be 1.0 or 1
# At this time, this will produce more similar output as example
pd.DataFrame(cout, columns=x.columns).to_csv("testout_2.2_1.csv", index=False, float_format="%.17g")

# Attempt to try to convert 1.0 to 0.99999...
# But failed...
# out = pd.DataFrame(cout, columns=x.columns)

# out_str = out.map(lambda v: format(v, ".17g") if pd.notna(v) else "")
# out_str.to_csv("testout_2.2.csv", index=False)