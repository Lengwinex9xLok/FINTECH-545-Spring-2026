import pandas as pd
import numpy as np

df = pd.read_csv("data/test1.csv")

complete = df.dropna(axis=0, how="any")

cov_mat = np.cov(complete.to_numpy(), rowvar=False,ddof=1)

out = pd.DataFrame(cov_mat, columns=df.columns)

out.to_csv("testout_1.1.csv", index=False, float_format="%.17g")
