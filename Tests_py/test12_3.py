import pandas as pd
from bt_american_discrete import bt_american_discrete

options = pd.read_csv("data/test12_3.csv")
options = options[options["ID"].notna()].copy()
options["ID"] = options["ID"].astype(int)

options["DividendDates"] = options["DividendDates"].apply(
    lambda s: [int(x.strip()) for x in str(s).split(",")]
)
options["DividendAmts"] = options["DividendAmts"].apply(
    lambda s: [float(x.strip()) for x in str(s).split(",")]
)

options["N"] = (options["DaysToMaturity"] * 2).astype(int)
options["DividendDates"] = options["DividendDates"].apply(
    lambda lst: [2 * x for x in lst]
)

outVals = []

for _, o in options.iterrows():
    val = bt_american_discrete(
        o["Option Type"] == "Call",
        float(o["Underlying"]),
        float(o["Strike"]),
        float(o["DaysToMaturity"]) / float(o["DayPerYear"]),
        float(o["RiskFreeRate"]),
        o["DividendAmts"],
        o["DividendDates"],
        float(o["ImpliedVol"]),
        int(o["N"])
    )
    outVals.append(val)

out_df = pd.DataFrame({
    "ID": options["ID"],
    "Value": outVals
})

out_df.to_csv("data/testout12_3_1.csv", index=False)