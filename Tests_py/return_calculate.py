import numpy as np
import pandas as pd

def return_calculate(prices, method="DISCRETE", dateColumn="date"):
    if dateColumn not in prices.columns:
        raise ValueError(f"dateColumn: {dateColumn} is notfound in dataFrame")
    vars_ = []
    for c in prices.columns:
        if c != dateColumn:
            vars_.append(c)
    
    p = prices[vars_].to_numpy(dtype=float)
    n, m = p.shape

# price ratios p[t+1] / p[t]
    ratios = p[1:, :] / p[:-1, :]

    meth = str(method).upper()
    if meth == "DISCRETE":
        p2 = ratios - 1.0
    elif meth == "LOG":
        p2 = np.log(ratios)
    else:
        raise ValueError ('method: {} must be in ("LOG","DISCRETE")'.format(method))
    
    dates = prices[dateColumn].iloc[1:]. reset_index(drop=True)

    data = {dateColumn: dates}
    for j, col in enumerate(vars_):
        data[col] = p2[:, j]
    
    out = pd.DataFrame(data)

    return out