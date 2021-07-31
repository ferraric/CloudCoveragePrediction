import xarray as xr
import time
import pandas as pd
import os
import sys

sys.path.append('../data_loader/')
from nwp_preprocessing import transform_nwp_data

INIT_HOUR = 0
DAY_STRIDE = 5


def subsample_nwp_data(
        data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    data = take_ensemble_mean_and_var(data)
    data = subsample_grid_points(data)
    data = change_time_dim_to_lead_time(data)
    return add_init_time_to_dimensions(data)


def take_ensemble_mean_and_var(
        data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    mean_clct = CLCT_mean = data.mean(dim='epsd_1').CLCT
    var_clct = CLCT_var = data.var(dim='epsd_1').CLCT
    data = data.drop('CLCT')
    data = data.drop('epsd_1')
    data = data.assign(CLCT_mean=mean_clct)
    data = data.assign(CLCT_var=var_clct)
    return data


def subsample_first_ensemble_member(
        data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    return data.isel(epsd_1=[0])


def subsample_grid_points(data: xr.core.dataset.Dataset,
                          x_stride=2,
                          y_stride=2) -> xr.core.dataset.Dataset:
    x_size = data.sizes['x_1']
    y_size = data.sizes['y_1']
    return data.isel(x_1=range(0, x_size, x_stride),
                     y_1=range(0, y_size, y_stride))


def add_init_time_to_dimensions(
        data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    return data.assign_coords(init_time=data.time.values[0]).expand_dims(
        dim='init_time')


def change_time_dim_to_lead_time(
        data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    return data.assign_coords(lead_time=range(121)).rename_dims(
        {'time': 'lead_time'})


all_transformed_predictions = None

start_date = pd.Timestamp(2014, 1, 1, INIT_HOUR)
end_date = pd.Timestamp(2018, 12, 31, INIT_HOUR)
time_step = pd.Timedelta(days=1)
init_time = start_date
skipped_dates = []
while init_time <= end_date:
    print("initialization time: ", init_time.strftime("%Y-%m-%d-%H"))
    assert init_time.hour == INIT_HOUR, "Initialization time {} should not be considered with INIT_HOUR {}".format(
        init_time, INIT_HOUR)

    start_timer = time.time()
    try:
        predictions = xr.open_mfdataset(
            "/mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_*" +
            init_time.strftime("%Y%m%d%H") + "*_CLCT.nc",
            preprocess=transform_nwp_data)
    except ValueError:
        print("multiple files for {} available, taking non reforecast".format(
            init_time))
        predictions = xr.open_mfdataset(
            "/mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_" +
            init_time.strftime("%Y%m%d%H") + "*_CLCT.nc",
            preprocess=transform_nwp_data)
    except OSError:
        print("no data available for {}, skipping to next day".format(
            init_time.strftime("%Y-%m-%d-%H")))
        skipped_dates.append(init_time)
        init_time += time_step
        continue
    print("--- Opening file took %s seconds ---" % (time.time() - start_timer))

    start_timer = time.time()
    transformed_predictions = subsample_nwp_data(predictions)
    print("--- Transformation took %s seconds ---" %
          (time.time() - start_timer))

    start_timer = time.time()
    if all_transformed_predictions is not None:
        all_transformed_predictions = xr.concat(
            [all_transformed_predictions, transformed_predictions],
            dim='init_time',
            join='override')
    else:
        all_transformed_predictions = transformed_predictions
    print("--- Concatenation took  %s seconds ---" %
          (time.time() - start_timer))

    init_time += time_step * DAY_STRIDE

print(all_transformed_predictions)
start_timer = time.time()
scratch_path = "/mnt/ds3lab-scratch/ferraric/nwp_subsampled_2x2_5_day_stride_mean_var"
full_output_path = os.path.join(
    scratch_path, "subsampled_CLCT_" + start_date.strftime("%Y-%m-%d-%H") +
    "_" + end_date.strftime("%Y-%m-%d-%H") + ".nc")
all_transformed_predictions.to_netcdf(path=full_output_path)
print("--- Saving file took  %s seconds ---" % (time.time() - start_timer))

print("skipped dates: ", skipped_dates)
