import pandas as pd
from return_calculate import return_calculate

prices = pd.read_csv("data/test6.csv")
rout = return_calculate(prices, method="LOG", dateColumn="Date")
rout.to_csv("data/testout6_2.csv", index=False, float_format="%.17g")