import pandas as pd
import numpy as np
import properscoring as ps
import time as t
from scipy.stats import norm
import pickle

model_name = "7_21_after_5000"
tic = t.clock()
data = pd.read_pickle("/mnt/ds3lab-scratch/yidai/real_predictions" + model_name + ".pkl")
print("Loading predictions:", t.clock() - tic, "seconds")
print("data.shape:", data.shape)

n = data.shape[0]
b = 1000000
predictions = ["pred_ensemble" + str(q) for q in range(21)]
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
    y_pred = df[predictions].values.clip(0, 100)
    tic2 = t.clock()
    crps = ps.crps_ensemble(y_true, y_pred)
    all_values = np.append(y_pred, y_true.reshape((-1, 1)), axis=-1)
    corner_cases_ind = (all_values == 0.0) | (all_values == 100.0)
    all_values[corner_cases_ind] = all_values[corner_cases_ind] + 0.001 * np.random.standard_normal(all_values[corner_cases_ind].shape)
    ranks = all_values.argsort(axis=-1).argsort(axis=-1)
    ranks = ranks[:, -1]
    lon_lat_pairs = [reversed_dict[id] for id in df.lat_lon_id.values]
    errors = pd.DataFrame({"init_time": df.real_init_time, "it": df.init_time, "time": df.time, "lon": [pair[0] for pair in lon_lat_pairs], "lat": [pair[1] for pair in lon_lat_pairs], 
                           "CRPS": crps, "ranks": ranks})
    errors["season"] = [season(it.strftime("%m")) for it in errors.init_time]
    errors.to_pickle("/mnt/ds3lab-scratch/yidai/evaluation/" + model_name + "/" + model_name + "_CRPS_cdf" + str(i) + "of" + str(n // b) + ".pkl")
    print("Batch", i, "of", n // b, "finished in", t.clock() - tic, "seconds", end=" " * 20 + "\n")


