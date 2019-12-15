import pandas as pd
import numpy as np
import properscoring as ps
import time as t
from scipy.stats import norm
import pickle

model_name = "mv_mv"
tic = t.clock()
data = pd.read_pickle("/mnt/ds3lab-scratch/yidai/real_predictions" + model_name + ".pkl")
# data = pd.read_pickle("/mnt/ds3lab-scratch/yidai/input_to_tf_q7lb_with_init_time_dummy_pred_mv.pkl")
print("Loading predictions:", t.clock() - tic, "seconds")
print("data.shape:", data.shape)

n = data.shape[0]
b = 1000000
no_quantiles = 21
reversed_dict = pd.read_pickle("/mnt/ds3lab-scratch/yidai/reversed_dict.pkl")

def season(month):
    if month in ["03", "04", "05"]:
        return "spring"
    if month in ["06", "07", "08"]:
        return "summer"
    if month in ["09", "10", "11"]:
        return "autumn"
    if month in ["12", "01", "02"]:
        return "winter"

for i in range(n // b + int(n % b != 0)): 
    tic = t.clock()
    if i < n // b:
        df = data.iloc[i * b : (i + 1) * b]
    else:
        df = data.iloc[i * b : n]
    y_true = df.labels.values
    mu = df.pred_mean.values.clip(0, 100)
    var = np.abs(df.pred_var.values)
    sigma = np.sqrt(var).clip(0.00001, np.inf)
    tic2 = t.clock()
    cdf_values = norm.cdf(y_true, loc=mu, scale=sigma)
    print("Computing cdf_values:", t.clock() - tic2, "seconds", end=" " * 10 + "\r")
    zero_values_ind = (y_true <= 0)
    cdf_values[zero_values_ind] = np.random.uniform(low=0, high=cdf_values[zero_values_ind], size=cdf_values[zero_values_ind].shape)
    hundred_values_ind = y_true >= 100
    cdf_values[hundred_values_ind] = np.random.uniform(low=cdf_values[hundred_values_ind], high=1, size=cdf_values[hundred_values_ind].shape)

    quantiles = np.broadcast_to([(i + 0.5) / no_quantiles for i in range(no_quantiles)], (mu.shape[0], no_quantiles))
    tic2 = t.clock()
    y_pred = norm.ppf(quantiles, loc=mu.reshape((-1, 1)), scale=sigma.reshape((-1, 1))).clip(0, 100)
    print("Computing 21 quantiles:", t.clock() - tic2, "seconds", end=" " * 10 + "\r")
    tic2 = t.clock()
    crps = ps.crps_ensemble(y_true, y_pred)
    print("Computing ensemble loss:", t.clock() - tic2, "seconds", end=" " * 10 + "\r")
    lon_lat_pairs = [reversed_dict[id] for id in df.lat_lon_id.values]
    errors = pd.DataFrame({"init_time": df.real_init_time, "it": df.init_time, "time": df.time, "lon": [pair[0] for pair in lon_lat_pairs], "lat": [pair[1] for pair in lon_lat_pairs], 
                           "CRPS": crps, "cdf_values": cdf_values})
    errors["season"] = [season(it.strftime("%m")) for it in errors.init_time]
    errors.to_pickle("/mnt/ds3lab-scratch/yidai/evaluation/" + model_name + "/" + model_name + "_CRPS_cdf" + str(i) + "of" + str(n // b) + ".pkl")
    print("Batch", i, "of", n // b, "finished in", t.clock() - tic, "seconds", end=" " * 20 + "\n")

