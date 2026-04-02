import pandas as pd
import numpy as np

stWgt = pd.read_csv("data/test11_2_weights.csv")['W'].to_numpy()
factor_returns_df = pd.read_csv("data/test11_2_factor_returns.csv")
stock_returns = pd.read_csv("data/test11_2_stock_returns.csv").to_numpy()
beta = pd.read_csv("data/test11_2_beta.csv").iloc[:, 1:].to_numpy()

factors = list(factor_returns_df.columns)
factor_returns = factor_returns_df.to_numpy()

n = stock_returns.shape[0]
n_stocks = stock_returns.shape[1]
n_factors = factor_returns.shape[1]

pReturn = np.zeros(n)
weights = np.zeros((n, n_stocks))
factorWeights = np.zeros((n, n_factors))
residReturn = np.zeros(n)
lastW = stWgt.copy()

for i in range(n):
    # Save Current Weights in Matrix
    weights[i, :] = lastW

    # Factor Weight
    factorWeights[i, :] = beta.T @ lastW

    # Update Weights by return
    lastW = lastW * (1.0 + stock_returns[i, :])

    # Portfolio return is the sum of the updated weights
    pR = sum(lastW)
    # Normalize the weights back so sum = 1
    lastW = lastW / pR
    # Store the return
    pReturn[i] = pR - 1

    # Residual
    residReturn[i] = pReturn[i] - factorWeights[i, :] @ factor_returns[i, :]

# Calculate the total return
totalRet = np.exp(sum(np.log(pReturn + 1))) - 1
# Calculate the Carino K
k = np.log(totalRet + 1) / totalRet
# Carino k_t is the ratio scaled by 1/K
carinoK = np.log(1.0 + pReturn) / pReturn / k
# Calculate the return attribution
attrib = factor_returns * factorWeights * carinoK[:, np.newaxis]
attrib_alpha = residReturn * carinoK

# Set up a DataFrame for output
Attribution = pd.DataFrame({'Value': ['TotalReturn', 'Return Attribution']})
# Loop over the factors
for j, s in enumerate(factors):
    # Total Factor return over the period
    tr = np.exp(sum(np.log(factor_returns[:, j] + 1))) - 1
    # Attribution Return
    atr = sum(attrib[:, j])
    Attribution[s] = [tr, atr]
Attribution['Alpha'] = [np.exp(sum(np.log(residReturn + 1))) - 1, sum(attrib_alpha)]
Attribution['Portfolio'] = [totalRet, totalRet]

# Realized Volatility Attribution

# Y is factor returns scaled by factor weights, plus residual
Y = np.column_stack([factor_returns * factorWeights, residReturn])
# Set up X with the Portfolio Return
X = np.column_stack([np.ones(n), pReturn])
# Calculate the Beta and discard the intercept
B = (np.linalg.inv(X.T @ X) @ X.T @ Y)[1, :]
# Component SD is Beta times the standard Deviation of the portfolio
cSD = B * np.std(pReturn, ddof=1)

# Add the Vol attribution to the output
vol_row = {'Value': 'Vol Attribution'}
for j, s in enumerate(factors):
    vol_row[s] = cSD[j]
vol_row['Alpha'] = cSD[-1]
vol_row['Portfolio'] = np.std(pReturn, ddof=1)
Attribution = pd.concat([Attribution, pd.DataFrame([vol_row])], ignore_index=True)

Attribution.to_csv("data/testout11_2.csv", index=False)
