import math

def bt_american(call, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = math.exp(ivol * math.sqrt(dt))
    d = 1.0 / u
    pu = (math.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = math.exp(-rf * dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)
    optionValues = [0.0] * nNodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            optionValues[idx] = max(0.0, z * (price - strike))

            if j < N:
                continuation = df * (
                    pu * optionValues[idxFunc(i + 1, j + 1)] +
                    pd * optionValues[idxFunc(i, j + 1)]
                )
                optionValues[idx] = max(optionValues[idx], continuation)

    return optionValues[0]

