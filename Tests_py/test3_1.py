import pandas as pd
from near_psd import near_psd

cin = pd.read_csv("data/testout_1.3.csv")
cout = near_psd(cin.to_numpy(), epsilon=0.0)

pd.DataFrame(cout, columns=cin.columns).to_csv("data/testout_3.1.csv", index=False, float_format="%.17g")
