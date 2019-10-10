import xarray as xr
import numpy as np

for year in [2014, 2016, 2017, 2018, 2019]:
	for month in range(1, 13, 1):
		for day_range in range(0, 4, 1):
			print("For year: ", year, "and month: ", month, "and days starting with: ", day_range)
			try:
				predictions = xr.open_mfdataset(
					"../../../../../mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_*?" + str(year) + str(month).zfill(2) + str(day_range) + "*_CLCT.nc")
			except OSError as e:	
				print(e.errno)
				continue
			total_number_of_values = predictions.CLCT.values.size
			print("total number of values: ", total_number_of_values)
			print("percentage of clct < 0: ")
			print(np.count_nonzero(predictions.CLCT.values < 0)/total_number_of_values)
			print("percentage of clct > 100: ")
			print(np.count_nonzero(predictions.CLCT.values > 100)/total_number_of_values)
			print("max clct: ")
			print(np.max(predictions.CLCT.values))
			print("min clct: ")
			print(np.min(predictions.CLCT.values))
			print()

