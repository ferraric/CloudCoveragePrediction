from comet_ml import Experiment
import json
import os
import random
import sys
sys.path.append('../')
from data_loader.data_generator_mean_var import DataGenerator
from models.mean_var_from_mv_quantiles import Model_mean_var
from trainers.crps_trainer_mv_to_mean_var import CRPSTrainer
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
    #print(config)
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

    model = Model_mean_var(config)
    data_input_shape = next(iter(data.train_data))[0].shape

    trainer = CRPSTrainer(model, data, config, experiment)
    trainer.train()


if __name__ == "__main__":
    main()
