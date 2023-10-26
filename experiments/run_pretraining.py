import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *
from source import *

import argparse
import torch
import random
import numpy
import time
import yaml

# python experiments/run_pretraining.py --dataset miniimagenet --model conv --num_ways 5 --num_shots 1 --seeds 0 --device cuda:0

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting the experiments directory for loading and saving.
directory = os.path.dirname(os.path.abspath(__file__)) + "/"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Parsing arguments to construct experiments.
# ============================================================

parser = argparse.ArgumentParser(description="Experiment Runner")

# Experiment settings.
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--seeds", required=True, type=int, nargs="+")
parser.add_argument("--device", required=False, type=str)

# Registering all optional configuration hyper-parameters.
register_configurations(parser)

# Retrieving the dictionary of arguments.
args, args_unknown = parser.parse_known_args()

if args.device is not None:
    device = args.device

if args.fast:  # Makes code non-deterministic (but faster).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ============================================================
# Constructing and executing experiments.
# ============================================================


def _run_experiment(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed_all(random_state)
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the custom dataset object.
    training, validation, testing = dataset(device=device, **config)

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "pretraining-" + args.dataset + "-" + args.model + "-" + \
                str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot-" + str(random_state)

    print("pretraining", args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    # Creating the base model, with.
    base_model = model(
        input_channels=config["input_channels"],
        track_running_stats=config["track_running_stats"],
        num_ways=training.dataset.num_classes
    ).to(device)

    # Creating the base model's *meta* optimizer.
    pretrain_optimizer = optimizer_archive[config["pretrain_optimizer_name"]](
        base_model.parameters(), **config["pretrain_optimizer_settings"])

    pretrain_scheduler = scheduler_archive[config["pretrain_scheduler_name"]](
        pretrain_optimizer, **config["pretrain_scheduler_settings"])

    base_model, meta_history, fine_tuning_history  = pretraining(
        base_model, pretrain_optimizer, pretrain_scheduler, training, validation,
        gradient_steps=config["pretrain_gradient_steps"],
        batch_size=config["pretrain_batch_size"],
        loss_function=objective_archive[config["task_loss_function"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        device=device, **config
    )

    # Recording the learning meta-data.
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Access the last layer of the base model.
    old_output_layer = list(base_model.modules())[-1]

    # Create a new dense/linear layer and replacing the last layer with the new layer.
    base_model.output_layer = torch.nn.Linear(old_output_layer.in_features, config["num_ways"]).to(device)
    
    # Initializing the head of the network.
    torch.nn.init.normal_(base_model.output_layer.weight, 0, 0.01)
    base_model.output_layer.bias.data.zero_()

    # Saving the pretrained model.
    export_model(base_model, res_directory, args.dataset + "-" + args.model + "-" +
                 str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot")

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Recording the training history.
    results["meta_history"] = meta_history
    results["fine_tuning_history"] = fine_tuning_history

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)

    print("pretraining", args.dataset, args.model, "seed", str(random_state), "complete")


# Opening the relevant configurations file.
with open(dataset_archive[args.dataset]["config"]) as file:
    config = yaml.safe_load(file)

required_args = {"dataset", "model", "seeds", "device"}
override_configurations(args, args_unknown, required_args, config)

# Retrieving the function for the selected dataset.
dataset_fn = dataset_archive[args.dataset]["data"]

# Retrieving the function for the selected model.
model_fn = model_archive[args.model]

# Executing the experiments with the given arguments.
for random_state in args.seeds:
    _run_experiment(dataset_fn, model_fn, config, random_state)
