import pandas as pd
from bt_american import bt_american
from finite_difference_gradient import finite_difference_gradient

options = pd.read_csv("data/test12_1.csv")
options = options[options["ID"].notna()].copy()
options["ID"] = options["ID"].astype(int)

outVals = [
    bt_american(
        row["Option Type"] == "Call",
        row["Underlying"],
        row["Strike"],
        row["DaysToMaturity"] / row["DayPerYear"],
        row["RiskFreeRate"],
        row["RiskFreeRate"] - row["DividendRate"],
        row["ImpliedVol"],
        500
    )
    for _, row in options.iterrows()
]

def fcall(parms):
    return bt_american(True, parms[0], parms[1], parms[2], parms[3], parms[4], parms[5], 500)

def fput(parms):
    return bt_american(False, parms[0], parms[1], parms[2], parms[3], parms[4], parms[5], 500)

deltas = []
gammas = []
vegas = []
rhos = []
thetas = []

for _, o in options.iterrows():
    parms = [
        o["Underlying"],
        o["Strike"],
        o["DaysToMaturity"] / o["DayPerYear"],
        o["RiskFreeRate"],
        o["RiskFreeRate"] - o["DividendRate"],
        o["ImpliedVol"]
    ]

    if o["Option Type"] == "Call":
        v = fcall(parms)
        grad = finite_difference_gradient(fcall, parms)

        deltas.append(grad[0])

        d = 1.5
        parms_up = parms.copy()
        parms_down = parms.copy()
        parms_up[0] += d
        parms_down[0] -= d
        gamma1 = fcall(parms_up)
        gamma2 = fcall(parms_down)
        gammas.append((gamma1 + gamma2 - 2 * v) / (d ** 2))

        vegas.append(grad[5])
        rhos.append(grad[3])
        thetas.append(grad[2])
    
    else:
        v = fput(parms)
        grad = finite_difference_gradient(fput, parms)

        deltas.append(grad[0])

        d = 1.5
        parms_up = parms.copy()
        parms_down = parms.copy()
        parms_up[0] += d
        parms_down[0] -= d
        gamma1 = fput(parms_up)
        gamma2 = fput(parms_down)
        gammas.append((gamma1 + gamma2 - 2 * v) / (d ** 2))

        vegas.append(grad[5])
        rhos.append(grad[3])
        thetas.append(grad[2])

out_df = pd.DataFrame({
    "ID": options["ID"],
    "Value": outVals,
    "Delta": deltas,
    "Gamma": gammas,
    "Vegas": vegas,
    "Rho": rhos,
    "Theta": thetas
})

out_df.to_csv("data/testout12_2.csv", index=False)