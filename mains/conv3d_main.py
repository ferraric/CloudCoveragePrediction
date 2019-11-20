from comet_ml import Experiment
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import sys
import numpy as np
sys.path.append('../')

from data_loader.conv3d_data_loader import DataGenerator
from models.ferraric_0_model import CNNModel
from models.conv3d_model import Conv3dModel
from models.ferraric_1_model import LocalModel, SimpleModel
from models.identity_model import IdentityModel
from trainers.example_trainer import ExampleTrainer
from trainers.crps_trainer import CRPSTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.arg import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )

    if config.use_comet_experiments:
        experiment_id = experiment.connection.experiment_id
    else:
        experiment_id = "local" + str(random.randint(1, 1000000))

    config.summary_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id), "summary/"
    )
    config.checkpoint_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id), "checkpoint/"
    )
    create_dirs([config.summary_dir, config.checkpoint_dir])
    print("...creating folder {}".format(config.summary_dir))

    with open(
        os.path.join(config.summary_dir, "config_summary.json"), "w"
    ) as json_file:
        json.dump(config, json_file)

    data = DataGenerator(config, experiment)

    dummy_model = Conv3dModel(config)
    iterator = iter(data.train_data)
    dummy_inputs, _ = next(iterator)
    dummy_model(dummy_inputs)
    model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    with open(model_architecture_path, "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        dummy_model.summary(print_fn=lambda x: fh.write(x + "\n"))
    dummy_model.summary()
    experiment.log_asset(model_architecture_path)

    #data_input_shape = next(iter(data.train_data))[0].shape
    #model.log_model_architecture_to(experiment, data_input_shape)

    model = Conv3dModel(config)
    trainer = CRPSTrainer(model, data, config, experiment)
    trainer.train()


if __name__ == "__main__":
    main()
