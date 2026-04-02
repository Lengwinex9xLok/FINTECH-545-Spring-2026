import pandas as pd
import numpy as np
from scipy.optimize import minimize

covar = pd.read_csv("data/test5_3.csv").to_numpy()
means = pd.read_csv("data/test10_3_means.csv")['Mean'].to_numpy()
rf = 0.04
n = covar.shape[0]

def sr(w):
    m = w @ means - rf
    s = np.sqrt(w @ covar @ w)
    return m / s

# Weights with boundry at 0
w0 = np.ones(n) / n
constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1.0}]
bounds = [(0, None)] * n

result = minimize(lambda w: -sr(w), w0, bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-8, 'maxiter': 3000})

w = result.x
pd.DataFrame({'W': w}).to_csv("data/testout10_3.csv", index=False)
