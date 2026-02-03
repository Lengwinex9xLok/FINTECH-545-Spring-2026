import pandas as pd
import numpy as np
import csv

df = pd.read_csv("data/test1.csv")

X = df.dropna(). to_numpy(dtype=np.float64)

corr = np.corrcoef(X, rowvar=False)

np.fill_diagonal(corr, 1.0)

out = pd.DataFrame(corr, columns=df.columns)

# or in this following form if don't mind 1 not 1.0
# out. to_csv("testout_1.2_1.csv", index=False, float_format="%.16g")

# ensure “1” to become "1.0".
def fmt(x: float) -> str:
    if x == 1.0:
        return "1.0"
    return format(x, ".16g")

with open("testout_1.2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(out.columns.tolist())
    for row in out.to_numpy():
        w.writerow([fmt(float(v)) for v in row])
