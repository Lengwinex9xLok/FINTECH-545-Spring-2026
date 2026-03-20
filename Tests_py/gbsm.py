import math
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class GBSM:
    value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    cRho: float

def gbsm(call: bool, underlying, strike, ttm, rf, b, ivol, include_greeks: bool = False) -> GBSM:
    d1 = (math.log(underlying / strike) + (b + ivol**2/2) * ttm) / (ivol * math.sqrt(ttm))
    d2 = d1 - ivol * math.sqrt(ttm)

    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    rho = 0.0
    cRho = 0.0

    if call:
        delta = math.exp((b - rf) * ttm) * norm.cdf(d1)
        value = underlying * delta - strike * math.exp(-rf * ttm) * norm.cdf(d2)
    else:
        delta = math.exp((b - rf) * ttm) * (norm.cdf(d1) - 1.0)
        value = strike * math.exp(-rf * ttm) * norm.cdf(-d2) - underlying * math.exp((b - rf) * ttm) * norm.cdf(-d1)

    if include_greeks:
        gamma = norm.pdf(d1) * math.exp((b - rf) * ttm) / (underlying * ivol * math.sqrt(ttm))
        vega = underlying * math.exp((b - rf) * ttm) * norm.pdf(d1) * math.sqrt(ttm)

        if call:
            theta = (
                -underlying * math.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * math.sqrt(ttm))
                - (b - rf) * underlying * math.exp((b - rf) * ttm) * norm.cdf(d1)
                - rf * strike * math.exp(-rf * ttm) * norm.cdf(d2)
            )

            rho = ttm * strike * math.exp(-rf * ttm) * norm.cdf(d2)
            cRho = ttm * underlying * math.exp((b - rf) * ttm) * norm.cdf(d1)

        else:
            theta = (
                -underlying * math.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * math.sqrt(ttm))
                + (b - rf) * underlying * math.exp((b - rf) * ttm) * norm.cdf(-d1)
                + rf * strike * math.exp(-rf * ttm) * norm.cdf(-d2)
            )

            rho = -ttm * strike * math.exp(-rf * ttm) * norm.cdf(-d2)
            cRho = -ttm * underlying * math.exp((b - rf) * ttm) * norm.cdf(-d1)
        
    return GBSM(value, delta, gamma, vega, theta, rho, cRho)