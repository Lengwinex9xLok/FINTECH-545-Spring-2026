import pandas as pd
from chol_psd import chol_psd

cin = pd.read_csv("data/testout_3.1.csv")
cout = chol_psd(cin.to_numpy())

pd.DataFrame(cout, columns=cin.columns).to_csv("data/testout_4.1.csv",index=False)



