import pandas as pd
import numpy as np
from scipy.stats import norm

cin = pd.read_csv("data/test7_1.csv")
cin_matrix = cin.dropna().to_numpy()
mu = cin_matrix.mean()
sigma = cin_matrix.std(ddof=1)
df = pd.DataFrame({"mu": [f"{mu:.18f}"], "sigma": [f"{sigma:.17f}"]})
df.to_csv("data/testout7_1.csv", index=False)