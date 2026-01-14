import pandas as pd
import numpy as np
from scipy.stats import t

cin = pd.read_csv("data/test7_2.csv")
cin_matrix = cin.dropna().to_numpy()
nu, mu, sigma = t.fit(cin_matrix)
df = pd.DataFrame({"mu": [mu], "sigma": [sigma], "nu": [nu]})
df.to_csv("data/testout7_2.csv", index=False)