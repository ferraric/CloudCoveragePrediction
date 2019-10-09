import os
import time
import datetime
import pathlib
import numpy as np
from pudb import set_trace

import pymchdwh as dwh
import xarray as xr
import keras
from keras import layers
from keras.callbacks import callbacks

from mlpp import data_handling, data_loading, preprocessing
from mlpp import models, custom_losses, custom_metrics


PUNKTDB_URL = 'http://zuenjx399.meteoswiss.ch:5001/api/point'
GITTERDB_URL = 'http://zuenjx399.meteoswiss.ch:5000/api/grid'


def train_model(configurations):
    # Extract configurations
    inputs = configurations['inputs']
    targets = configurations['targets']
    n_inputs = len(inputs)
    n_outputs = len(targets)
    loss = configurations['loss']

    # Setup model
    model = models.shallow_mlp(n_inputs, n_outputs)
    model.compile(
        optimizer='adam',
        loss=loss
    )

    # Get data generators
    generators = data_handling.get_generators(
        configurations['reftimes'],
        configurations['stations'],
        inputs,
        targets,
        configurations['leadtimes'],
        configurations['members'],
        configurations['datasplit'],
        configurations['batch_size']
    )

    # Create logging directory
    logpath = pathlib.Path(configurations['logpath'])
    try:
        current_time = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
        current_logpath = logpath / current_time
        os.mkdir(current_logpath)
    except OSError:
        raise OSError("Could not create log directory")

    # Callbacks
    checkpoints = callbacks.ModelCheckpoint(current_logpath)
    early_stopping = callbacks.EarlyStopping()
    tensorboard = callbacks.tensorboard_v1.TensorBoard(
        log_dir=(current_logpath / 'tensorboard').as_posix()
    )
    callback_list = [checkpoints, early_stopping, tensorbard]

    # Fit model
    history = model.fit_generator(
        generators[0],
        epochs=20,
        callbacks=callback_list,
        validation_data=generators[1],
        use_multiprocessing=False,
        shuffle=True
    )

    # Evaluate on test set
    test_loss = model.evaluate_generator(generators[2])
    print(f"Test loss is {test_loss}")


def get_configurations():
    datasplit = [0.8, 0.1, 0.1]
    batch_size = 128
    leadtimes = list(range(1, 121))
    members = list(range(21))

    # Define reftimes
    start_time = datetime.datetime(2019, 9, 1, 0, 0)
    end_time = datetime.datetime(2019, 10, 9, 12, 0)
    reftimes = data_handling.get_reftimes(start_time, end_time, 12)

    # Load stations
    stations_path = pathlib.PurePosixPath(
        '/prod/zue/fc_development/users/hes/PostprocVeri/hux/data/preliminary_list_of_stations.json'
    )
    stations = data_loading.load_station_list(stations_path)
    stations = preprocessing.clean_station_list(stations)

    # Logging path
    logpath = '/prod/zue/fc_development/users/hux/python_packages/mlpp/logs'

    # Inputs and targets
    inputs = ['dayofyear', 'hourofday', 'lon', 'lat', 'leadtime']
    targets = ['eastward_wind']

    # Losses and metrics
    loss = 'mse'
    metrics = [
        'mae',
        'acc',
        custom_metrics.crps_gaussian
    ]

    configurations = {
        'inputs': inputs,
        'targets': targets,
        'reftimes': reftimes,
        'stations': stations,
        'leadtimes': leadtimes,
        'members': members,
        'datasplit': datasplit,
        'batch_size': batch_size,
        'logpath': logpath,
        'loss': loss,
        'metrics': metrics
    }

    return configurations


if __name__ == '__main__':
    configurations = get_configurations()
    train_model(configurations)
