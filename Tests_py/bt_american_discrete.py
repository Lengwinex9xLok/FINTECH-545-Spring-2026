import math
from bt_american import bt_american


def bt_american_discrete(call, underlying, strike, ttm, rf, divAmts, divTimes, ivol, N):
    if len(divAmts) == 0 or len(divTimes) == 0:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)
    elif divTimes[0] > N:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)
    
    dt = ttm / N
    u = math.exp(ivol * math.sqrt(dt))
    d = 1.0 / u
    pu = (math.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = math.exp(-rf * dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)
    
    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i
    
    nDiv = len(divTimes)
    first_div_time = divTimes[0]
    nNodes = nNodeFunc(first_div_time)

    optionValues = [0.0] * nNodes

    for j in range(first_div_time, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))

            if j < first_div_time:
                exercise = max(0.0, z * (price - strike))
                continuation = df * (
                    pu * optionValues[idxFunc(i + 1, j + 1)] +
                    pd * optionValues[idxFunc(i, j + 1)]
                )
                optionValues[idx] = max(exercise, continuation)
            else:
                ValNoExercise = bt_american_discrete(
                    call,
                    price - divAmts[0],
                    strike,
                    ttm - first_div_time * dt,
                    rf,
                    divAmts[1:nDiv],
                    [t - first_div_time for t in divTimes[1:nDiv]],
                    ivol,
                    N - first_div_time
                )
                valExercise = max(0.0, z * (price - strike))
                optionValues[idx] = max(ValNoExercise, valExercise)

    return optionValues[0]











