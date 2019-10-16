import numpy as np


def return_quantiles(input_arr, max_quantiles):
    percentiles = []
    for i in range(1, max_quantiles + 1):
        percentiles.append(((i - 0.5) / max_quantiles))
    q = np.percentile(input_arr, percentiles, axis=0)
    return np.array(q)


#example usecase
input_arr = np.random.randn(419, 5550)
max_quantiles = 21
f = return_quantiles(input_arr, max_quantiles)
print(f.shape)
print(f)
