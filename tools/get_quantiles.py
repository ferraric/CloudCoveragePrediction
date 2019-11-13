import numpy as np

def return_quantiles(input_arr: np.ndarray, max_quantiles: int) -> np.ndarray:
    quantiles = []
    for i in range(1, max_quantiles + 1):
        quantiles.append(((i - 0.5) / max_quantiles))
    return np.quantile(input_arr, quantiles, axis=0)

#example usecase
input_arr = np.arange(100)
max_quantiles = 21
f = return_quantiles(input_arr, max_quantiles)
print(f.shape)
print(f)
