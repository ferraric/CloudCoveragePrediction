import pandas as pd
import os

folder = os.path.join("~", "evaluation")
dates = pd.date_range("20140101 00", "20181226 12", freq="12H")
file_names = ["persistence_CRPS_" + date.strftime("%Y%m%d%H") + ".pkl" for date in dates]
for name in file_names:
    if not os.path.exists(name):
        print(name)
