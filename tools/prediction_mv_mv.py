import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
import properscoring as ps
import time as t
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

lookup_ids = pd.read_pickle("/mnt/ds3lab-scratch/yidai/interpolation_lookup_ids.pkl")
def nearest_neighbors(lon, lat, k = 4): 
    return lookup_ids[(lat, lon)]

tic = t.clock()
data = pd.read_pickle("/mnt/ds3lab-scratch/yidai/input_to_tf_labels_with_init_time.pkl")  # mv
# data = pd.read_pickle("/mnt/ds3lab-scratch/yidai/input_to_tf_q7lb_with_init_time_subset.pkl")  # 7q
print("Reading dataframe:", t.clock() - tic, "seconds")
meta_info = ["month", "hour", "init_time", "lat_lon_id", "lead_time"]
features = ["CLCT_mean", "CLCT_var"]  # mv
# features = ["CLCT_quantile" + str(q) for q in range(7)]  # 7q
features += meta_info
tic = t.clock()
# model = tf.keras.models.load_model("/mnt/ds3lab-scratch/srhea/code/ml-pipeline/mains/dummy_models/dummy1")
model_name = "mv_mv"  # 
model = tf.keras.models.load_model("/mnt/ds3lab-scratch/srhea/code/ml-pipeline/mains/" + model_name)
print("Loading model:", t.clock() - tic, "seconds")
b = 8192
predictions = ["pred_mean", "pred_var"]  # out mv
# predictions = ["pred_quantile" + str(q) for q in range(21)]  # out 21
p = len(predictions)
tic = t.clock()
full_input = tf.convert_to_tensor(data[features].values, dtype=tf.float64)
print("Converting to tensor:", t.clock() - tic, "seconds")
n = full_input.shape[0]
full_output = np.zeros((n, p))
tic = t.clock()
for i in range(n // b): 
    full_output[i * b : (i + 1) * b, :] = np.array(model(tf.reshape(full_input[i * b : (i + 1) * b, :], (b, -1)))).reshape((b, p))
    print("Batch", i, "of", n // b, "done after", t.clock() - tic, "seconds", end=" " * 10 + "\r")
print("Compute outputs batch by batch:", t.clock() - tic, "seconds")
tic = t.clock()
for j in range(n // b * b, n): 
    model_input = np.repeat(np.array(full_input[j, :]).reshape((1, -1)), b, axis=0)
    model_output = model(tf.convert_to_tensor(model_input, dtype=tf.float64))
    full_output[j, :] = np.concatenate([t for t in model_output], axis=1)[0, :]  # out mv
    # full_output[j, :] = np.array(model_output)[0, :]  # out 21
    print("Leftover", j - n // b * b, "of", n - n // b * b, "done after", t.clock() - tic, "seconds", end=" " * 10 + "\r")
print(full_output)
print("Compute leftover outputs:", t.clock() - tic, "seconds")
tic = t.clock()
for key in predictions: 
    data[key] = np.zeros(n)
data[predictions] = full_output
print("Saving output into dataframe:", t.clock() - tic, "seconds")
data.to_pickle("/mnt/ds3lab-scratch/yidai/real_predictions" + model_name + ".pkl")



