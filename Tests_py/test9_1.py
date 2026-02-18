import numpy as np
import pandas as pd
from scipy.stats import norm, t

from fit_normal import fit_normal
from neg_T_loglikelihood import neg_T_loglikelihood
from fit_general_t import fit_general_t
from simulate_pca import simulate_pca
from VaR_a import VaR
from ES_a import ES

# aggRisk(values, [:Stock])
def aggRisk(values, alpha=0.05):
    out = []
    for stock, g in values.groupby("Stock"):
        pnl = g["pnl"].to_numpy(dtype=float)
        cv = float(g["currentValue"].iloc[0])

        VaR95 = VaR(pnl, alpha)
        ES95 = ES(pnl, alpha)

        out.append({
            "Stock": stock,
            "VaR95": VaR95,
            "ES95": ES95,
            "VaR95_Pct": VaR95 / cv,
            "ES95_Pct": ES95 / cv,
        })

    total_pnl_by_iter = values.groupby("iteration", as_index=False)["pnl"].sum()
    total_pnl = total_pnl_by_iter["pnl"].to_numpy(dtype=float)
    total_cv = float(values.drop_duplicates("Stock")["currentValue"].sum())

    VaR95 = VaR(total_pnl, alpha)
    ES95 = ES(total_pnl, alpha)

    out.append({
        "Stock": "Total",
        "VaR95": VaR95,
        "ES95": ES95,
        "VaR95_Pct": VaR95 / total_cv,
        "ES95_Pct": ES95 / total_cv,
    })
    return pd.DataFrame(out)

# Read
cin = pd.read_csv("data/test9_1_returns.csv")
port_raw = pd.read_csv("data/test9_1_portfolio.csv")
portfolio = pd.DataFrame({
    "Stock": port_raw["Stock"].astype(str),
    "currentValue": port_raw["Holding"].astype(float)
    * port_raw["Starting Price"].astype(float),
})

# models["A"] = fit_normal(cin.A)
muA, sigA = fit_normal(cin["A"].to_numpy())

# models["B"] = fit_general_t(cin.B)
muB, sB, nuB = fit_general_t(cin["B"].to_numpy())

nSim = 10000

# U = [models["A"].u models["B"].u]
uA = norm.cdf(cin["A"].to_numpy(), loc=muA, scale=sigA)
uB = t.cdf(cin["B"].to_numpy(), df=nuB, loc=muB, scale=sB)
U = np.column_stack([uA, uB])

# spcor = corspearman(U)
spcor = pd.DataFrame(U).corr(method="spearman").to_numpy()

# uSim = simulate_pca(spcor,nSim)
# uSim = cdf.(Normal(),uSim)
u_Sim = simulate_pca(spcor, nSim)
uSim = norm.cdf(u_Sim)

# simRet = DataFrame(:A=>models["A"].eval(uSim[:,1]), :B=>models["B"].eval(uSim[:,2]))
simA = norm.ppf(uSim[:, 0], loc=muA, scale=sigA)
simB = t.ppf(uSim[:, 1], df=nuB, loc=muB, scale=sB)
simRet = pd.DataFrame({"A": simA, "B": simB})

iteration = pd.DataFrame({"iteration": np.arange(1, nSim + 1)})
values = portfolio.merge(iteration, how="cross")

idx = values["iteration"].to_numpy() - 1
stk = values["Stock"].to_numpy()
cuv = values["currentValue"].to_numpy(dtype=float)
ret = np.where(stk == "A", simRet["A"].to_numpy()[idx], simRet["B"].to_numpy()[idx])

simulatedValue = cuv * (1.0 + ret)
pnl = simulatedValue - cuv

values["pnl"] = pnl
values["simulatedValue"] = simulatedValue

# risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])
risk = aggRisk(values, alpha=0.05)[["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]]

risk.to_csv("testout9_1.csv", index=False, float_format="%.17g")
