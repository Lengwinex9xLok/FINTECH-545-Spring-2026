def finite_difference_gradient(func, x):
    x = list(map(float, x))
    grad = []

    for k in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[k]))
        x_up = x.copy()
        x_down = x.copy()
        x_up[k] += h
        x_down[k] -= h
        grad.append((func(x_up) - func(x_down)) / (2 * h))
    
    return grad