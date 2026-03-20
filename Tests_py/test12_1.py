import pandas as pd
from gbsm import gbsm

options = pd.read_csv("data/test12_1.csv")
options = options[options["ID"].notna()].copy()
options["ID"] = options["ID"].astype(int)

outVals = [
    gbsm(
        call=(row["Option Type"] == "Call"),
        underlying=row["Underlying"],
        strike=row["Strike"],
        ttm=row["DaysToMaturity"] / row["DayPerYear"],
        rf=row["RiskFreeRate"],
        b=row["RiskFreeRate"] - row["DividendRate"],
        ivol=row["ImpliedVol"],
        include_greeks=True
    )
    for _, row in options.iterrows()
]

values = [v.value for v in outVals]
deltas = [v.delta for v in outVals]
gammas = [v.gamma for v in outVals]
vegas = [v.vega for v in outVals]
rhos = [v.rho for v in outVals]
thetas = [v.theta for v in outVals]

out_df = pd.DataFrame({
    "ID": options["ID"],
    "value": values,
    "Delta": deltas,
    "Gamma": gammas,
    "Vega": vegas,
    "Rho": rhos,
    "Theta": thetas
})

out_df.to_csv("data/testout12_1S.csv", index=False)