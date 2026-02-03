import pandas as pd
from higham_near_psd import higham_near_psd

cin = pd.read_csv("data/testout_1.3.csv")
cout = higham_near_psd(cin.to_numpy(), epsilon=1e-9, maxIter=100, tol=1e-9)

pd.DataFrame(cout, columns=cin.columns).to_csv("data/testout_3.3.csv", index=False, float_format="%.17g")
