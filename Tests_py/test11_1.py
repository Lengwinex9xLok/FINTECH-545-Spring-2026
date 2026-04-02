import pandas as pd
import numpy as np

returns = pd.read_csv("data/test11_1_returns.csv")
stWgt = pd.read_csv("data/test11_1_weights.csv")['W'].to_numpy()

stocks = list(returns.columns)
matReturns = returns.to_numpy()
n = matReturns.shape[0]

pReturn = np.zeros(n)
weights = np.zeros_like(matReturns)
lastW = stWgt.copy()

for i in range(n):
    # Save Current Weights in Matrix
    weights[i, :] = lastW

    # Update Weights by return
    lastW = lastW * (1.0 + matReturns[i, :])

    # Portfolio return is the sum of the updated weights
    pR = sum(lastW)
    # Normalize the weights back so sum = 1
    lastW = lastW / pR
    # Store the return
    pReturn[i] = pR - 1

# Calculate the total return
totalRet = np.exp(sum(np.log(pReturn + 1))) - 1
# Calculate the Carino K
k = np.log(totalRet + 1) / totalRet
# Carino k_t is the ratio scaled by 1/K
carinoK = np.log(1.0 + pReturn) / pReturn / k
# Calculate the return attribution
attrib = matReturns * weights * carinoK[:, np.newaxis]

# Set up a DataFrame for output
Attribution = pd.DataFrame({'Value': ['TotalReturn', 'Return Attribution']})
# Loop over the stocks
for j, s in enumerate(stocks):
    # Total Stock return over the period
    tr = np.exp(sum(np.log(matReturns[:, j] + 1))) - 1
    # Attribution Return
    atr = sum(attrib[:, j])
    Attribution[s] = [tr, atr]
Attribution['Portfolio'] = [totalRet, totalRet]

# Realized Volatility Attribution

# Y is our stock returns scaled by their weight at each time
Y = matReturns * weights
# Set up X with the Portfolio Return
X = np.column_stack([np.ones(n), pReturn])
# Calculate the Beta and discard the intercept
B = (np.linalg.inv(X.T @ X) @ X.T @ Y)[1, :]
# Component SD is Beta times the standard Deviation of the portfolio
cSD = B * np.std(pReturn, ddof=1)

# Add the Vol attribution to the output
vol_row = {'Value': 'Vol Attribution'}
for j, s in enumerate(stocks):
    vol_row[s] = cSD[j]
vol_row['Portfolio'] = np.std(pReturn, ddof=1)
Attribution = pd.concat([Attribution, pd.DataFrame([vol_row])], ignore_index=True)

Attribution.to_csv("data/testout11_1.csv", index=False)
