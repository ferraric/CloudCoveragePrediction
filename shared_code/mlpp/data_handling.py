import datetime
import math
import random
from pudb import set_trace

import xarray as xr
from keras.utils import Sequence

from mlpp import preprocessing
from mlpp import data_loading


class DataGenerator(Sequence):
    """
    Class that creates batches of training data from medium size datasets.
    """
    def __init__(
        self,
        dataset,
        reftimes,
        stations,
        inputs,
        targets,
        leadtimes,
        batch_size,
        target_mode='error',
        scaler=None,
        shuffle=True
    ):
        self.dataset = dataset
        self.reftimes = reftimes
        self.stations = stations
        self.inputs = inputs  # ', '.join(inputs)
        self.targets = targets  # ', '.join(outputs)
        self.leadtimes = leadtimes
        self.batch_size = min(batch_size, len(reftimes))
        self.target_mode = target_mode
        self.scaler = scaler
        self.shuffle = shuffle

    def __len__(self):
        # Return length of seqeuence of batches
        return math.ceil(len(self.ref_times) / self.batch_size)

    def __getitem__(self, idx):
        # Define batch
        batch_reftimes = self.reftimes[
            idx * self.batch_size:
            (idx + 1) * self.batch_size
        ]

        if len(batch_reftimes) == 0:
            raise ValueError('Empty batch!')

        if self.scaler is not None:
            dataset = self.scaler.transform(
                self.dataset.sel(reftime=batch_reftimes)
            )
        else:
            dataset = self.dataset.sel(reftime=batch_reftimes)

        # Get batch of data
        batch_x = self._get_inputs(dataset)

        # Get targets of model
        batch_y = self._get_targets(dataset, batch_reftimes)

        batch_x = batch_x.T
        batch_y = batch_y.T

        return batch_x, batch_y

    def _get_inputs(self, dataset):
        # Transform batch data to numpy array
        inputs = ['data.' + feature for feature in self.inputs]
        input_set = dataset[inputs]

        # Stack coordinates
        input_set = input_set.to_array()
        input_set = input_set.stack(
            sample=[
                'reftime',
                'leadtime',
                'station',
                'member'
            ]
        )

        return input_set.values

    def _get_targets(self, dataset, reftimes):
        # Transform batch data to numpy array
        target_set = xr.Dataset()
        for target in self.targets:
            name = "data." + target
            if self.target_mode == 'error':
                if target == 'northward_wind':
                    observations = preprocessing.get_wind_observations(
                        self.stations,
                        reftimes,
                        self.leadtimes,
                        'northward_wind'
                    )
                elif target == 'eastward_wind':
                    observations = preprocessing.get_wind_observations(
                        self.stations,
                        reftimes,
                        self.leadtimes,
                        'eastward_wind'
                    )

                # Broadcast together with member dimensions
                observations = observations.broadcast_like(dataset.member)

                # Scale observations if scaler provided
                if self.scaler is not None:
                    observations = self.scaler.transform(observations)

                # Form error
                model = dataset[name]
                target_set[name] = observations - model
            else:
                target_set[name] = dataset[name]

        # Get variables
        target_set = target_set.to_array()
        target_set = target_set.stack(
            sample=[
                'reftime',
                'leadtime',
                'station',
                'member'
            ]
        )
        return target_set.values

    def on_epoch_end(self):
        # Shuffle if option is set
        if self.shuffle:
            random.shuffle(self.reftimes)
            random.shuffle(self.stations)
        self.dataset = self.dataset.sel(stations=self.stations)


def get_reftimes(start_time, end_time, step):
    # Create a list of times between 'start_time' and 'end_time' with step size
    # 'step'
    dt = datetime.timedelta(hours=step)

    times = []
    time = start_time
    while time <= end_time:
        times.append(time)
        time += dt

    return times


def split_list(data, split, shuffle=False):
    if sum(split) != 1:
        raise ValueError("Splits don't add up to 1!")

    if shuffle:
        random.shuffle(data)

    n_data = len(data)
    split_data = []
    fract = 0
    for i in range(len(split)):
        start = int(fract*n_data)
        fract += split[i]
        end = int(fract*n_data)
        split_data.append(data[start:end])

    return split_data


def get_generators(
    reftimes,
    stations,
    inputs,
    targets,
    leadtimes=[4],
    members=list(range(21)),
    data_split=[0.8, 0.1, 0.1],
    batch_size=100
):
    """
    Provide data generators for train, validation and test data
    """
    requested_data = {
        'reftimes': reftimes,
        'stations': stations,
        'leadtimes': leadtimes,
        'members': members
    }

    # Split data into train, validation and test set
    reftimes_split = split_list(reftimes, data_split, shuffle=True)
    stations_split = split_list(stations, data_split, shuffle=True)

    # if loading_mode == 'database':
    #     ds = data_loading.load_from_database(targets, requested_data, 'database')
    # else:
    zarr_path = '/prod/zue/fc_development/users/hux/data/cosmoe_test/wind.zarr'
    ds = data_loading.load_from_zarr(zarr_path, requested_data)

    # Clean dataset from unnecessary variables and define standardizer
    ds, standardizer = preprocessing.preprocess_dataset(
        ds,
        inputs,
        targets,
        stations,
        reftimes,
        leadtimes,
        reftimes_split[0],
        stations_split[0]
    )
    set_trace()

    print("Creating generator instances")
    generators = []
    for i in range(3):
        generators.append(
            DataGenerator(
                ds,
                reftimes_split[i],
                stations_split[i],
                inputs,
                targets,
                leadtimes,
                members,
                batch_size,
                scaler=standardizer
            )
        )

    return generators
