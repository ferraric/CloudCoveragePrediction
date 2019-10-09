import time
import json

import requests
import xarray as xr


GITTERDB_URL = 'http://zuenjx399.meteoswiss.ch:5000/api/grid'


def load_dataset(targets, requested_data, mode):
    if mode == 'database':
        ds = load_from_database(targets, requested_data)

    return ds


def load_from_database(targets, requested_data):
    # We need to iterate over different requests, as the maximum duration
    # of a request is limited to 6 minutes
    reftimes = requested_data['reftimes']
    leadtimes = requested_data['leadtimes']
    chunk_size = 1

    print("Getting dataset information")
    timing_total_start_time = time.time()
    ds = None
    for i in range(0, len(reftimes), chunk_size):
        for target in targets:
            print("Making request...")
            timing_start_time = time.time()
            # Define requested parameters
            search_params = dict(
                parameter=target,
                model='COSMO-E',
                reftime=','.join(
                    [time.isoformat() for time in reftimes[i:i+chunk_size]]
                ),
                leadtime=','.join([str(t) for t in leadtimes])
            )

            search_request = requests.get(GITTERDB_URL, params=search_params)
            timing_end_time = time.time()
            print(f"Request took {(timing_end_time - timing_start_time):0.2f} seconds to complete")

            print("Loading dataset into xarray...")
            # Load data into xarray dataset
            timing_start_time = time.time()
            if ds is None:
                ds = xr.open_dataset(search_request.json()['opendap'])
            else:
                ds = xr.concat(
                    [ds, xr.open_dataset(search_request.json()['opendap'])],
                    'data'
                )
            timing_end_time = time.time()
            print(f"Loading took {(timing_end_time - timing_start_time):0.2f} seconds to complete")
            print(f"Progress {min(int((i+chunk_size)/len(reftimes)*100), 100)}%")
    print(f"Total duration was {timing_end_time - timing_total_start_time} seconds.")

    return ds


def load_from_zarr(path, requested_data):
    """
    Loads the data from a path to a local directory with zarr files. Zarr files
    are only usable with xarray if they have been written through xarray.
    """
    ds = xr.open_zarr(path)

    return ds


def load_station_list(path):
    with open(path, "r") as read_file:
        stations = json.load(read_file)

    return stations
