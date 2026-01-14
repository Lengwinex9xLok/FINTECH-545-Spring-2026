import pandas as pd
import numpy as np
#special is imported for gammaln
# initially imported t and try to do t.fit but not accurate enough compared with sample output
# So I tried this method in order to fit the output from Julia
from scipy.stats import t
from scipy import  special, optimize

cin = pd.read_csv("data/test7_2.csv")
cin_matrix = cin.dropna().to_numpy()

def neg_T_loglikelihood(mu, s, nu, x): # This function was reference from class (Week02/test.jl)
    n = x.size
    np12 = (nu + 1.0) / 2.0
    mess = (special.gammaln(np12) - special.gammaln(nu / 2.0) - np.log(np.sqrt(np.pi * nu) * s))
    xm = ((x - mu) / s) ** 2 * (1.0 / nu) + 1.0
    inner_sum = np.sum(np.log(xm))
    ll = n * mess - np12 * inner_sum
    return -ll


mu0 = float(cin_matrix.mean())
sigma0 = float(cin_matrix.std(ddof=0))
nu0 = 10.0
theta_0 = np.array([mu0, np.log(sigma0), np.log(nu0)])

#This is reference from ChatGPT with prompt: "How to do maximum MLE in t-distribution log likelihood."
def objective(theta): #Since optimize.minimize only support one argument
    mu = theta[0]
    s = np.exp(theta[1])
    nu = np.exp(theta[2])
    return neg_T_loglikelihood(mu, s, nu, cin_matrix)

# use numerically maximization of loglikelihood (same as minimize -ll)
res = optimize.minimize(
    objective,
    x0=theta_0,
    method="Nelder-Mead",#Nelder-Mead for increase convergence accuracy
    options={"maxiter": 200000, "xatol": 1e-14, "fatol": 1e-14},
)

mu_hat, log_s_hat, log_nu_hat = res.x
sigma_hat = float(np.exp(log_s_hat))
nu_hat = float(np.exp(log_nu_hat))

df = pd.DataFrame({"mu": [f"{mu_hat:.17f}"],"sigma": [f"{sigma_hat:.17f}"],"nu": [f"{nu_hat:.15f}"]})
df.to_csv("data/testout7_2.csv", index=False)