import pandas as pd
import numpy as np
from scipy import special, optimize
#import statsmodels.api as sm
from scipy.stats import t

cin = pd.read_csv("data/test7_3.csv").dropna()
y = cin["y"].to_numpy()
X = cin.drop(columns=["y"]).to_numpy()

# The following is OLS regression + t.fit (I tried at first but with large error)
#OLS regression where beta = (X'X) ^{-1}(X'y)
x_int = np.column_stack([np.ones(len(y)),X]) #[1, X]
#OLS beta = np.linalg.lstsq(x_int, y, rcond=None)[0] #beta
#Alpha, B1, B2, B3 = beta

#e = y - x_int @ beta
#nu, mu_fixed0, sigma = t.fit(e, floc=0.0) #mu = 0.0

# The following is based on test7_2_optimize.py, added t regression. 
n, p = x_int.shape

def neg_T_loglikelihood(mu, s, nu, x): # This function was reference from class (Week02/test.jl)
    n = x.size
    np12 = (nu + 1.0) / 2.0
    mess = (special.gammaln(np12) - special.gammaln(nu / 2.0) - np.log(np.sqrt(np.pi * nu) * s))
    xm = ((x - mu) / s) ** 2 * (1.0 / nu) + 1.0
    inner_sum = np.sum(np.log(xm))
    ll = n * mess - np12 * inner_sum
    return -ll

def neg_T_regression_loglikelihood(pa, y, x):
    p = x.shape[1]
    beta = pa[:p]
    sigma = np.exp(pa[p])
    nu = np.exp(pa[p+1])
    e = y - x @ beta
    return neg_T_loglikelihood(0.0, sigma, nu, e)

def objective(pa):#set in one argument
    return neg_T_regression_loglikelihood(pa, y, x_int)

beta0 = np.zeros(p)
sigma0 = float(np.std(y))
nu0 = 5.0

pa0 = np.r_[beta0, np.log(sigma0), np.log(nu0)]

res = optimize.minimize(
    objective,
    x0=pa0,
    method="Nelder-Mead",
    options={"maxiter": 200000, "xatol": 1e-14, "fatol": 1e-14}
)

pa_hat = res.x
beta_hat = pa_hat[:p]
sigma_hat = float(np.exp(pa_hat[p]))
nu_hat = float(np.exp(pa_hat[p + 1]))

Alpha,B1,B2,B3 = beta_hat

df = pd.DataFrame({"mu": [0.0], "sigma": f"{sigma_hat:.18f}", "nu": f"{nu_hat:.15f}", "Alpha": f"{Alpha:.17f}", "B1": f"{B1:.16f}", "B2": f"{B2:.16f}", "B3": f"{B3:.15f}"})
df.to_csv("data/testout7_3.csv", index=False)
