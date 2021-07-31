from comet_ml import Experiment
import json
import os
import random
import sys

sys.path.append('../')
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from data_loader.data_generator_7_quantiles import DataGenerator
from models.model_7_to_21 import Model_7_to_21
from trainers.crps_trainer_7_to_21_after_5000 import CRPSTrainer
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
        "../experiments", os.path.join(config.exp_name, experiment_id),
        "summary/")
    config.checkpoint_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id),
        "checkpoint/")
    create_dirs([config.summary_dir, config.checkpoint_dir])
    print("...creating folder {}".format(config.summary_dir))

    with open(os.path.join(config.summary_dir, "config_summary.json"),
              "w") as json_file:
        json.dump(config, json_file)

    data = DataGenerator(config, experiment)

    model = Model_7_to_21(config)
    data_input_shape = next(iter(data.train_data))[0].shape

    trainer = CRPSTrainer(model, data, config, experiment)
    trainer.train()


if __name__ == "__main__":
    main()
