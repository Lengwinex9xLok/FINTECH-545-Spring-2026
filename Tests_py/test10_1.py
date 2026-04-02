import pandas as pd
import numpy as np
from scipy.optimize import minimize

covar = pd.read_csv("data/test5_2.csv").to_numpy()
n = covar.shape[0]

# Function for Portfolio Volatility
def pvol(w):
    return np.sqrt(w @ covar @ w)

# Function for Component Standard Deviation
def pCSD(w):
    pVol = pvol(w)
    csd = w * (covar @ w) / pVol
    return csd

# Sum Square Error of cSD
def sseCSD(w):
    csd = pCSD(w)
    mCSD = sum(csd) / n
    dCsd = csd - mCSD
    se = dCsd * dCsd
    return 1.0e5 * sum(se)

# Weights with boundry at 0
w0 = np.ones(n) / n
constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
bounds = [(0, None)] * n

result = minimize(sseCSD, w0, bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-8, 'maxiter': 3000})

w = result.x
pd.DataFrame({'W': w}).to_csv("data/testout10_1.csv", index=False)
