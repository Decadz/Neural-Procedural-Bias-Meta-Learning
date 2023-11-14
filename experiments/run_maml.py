import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *
from source import *

import functools
import argparse
import torch
import random
import numpy
import time
import yaml

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

    # If we are using a pretrained backbone. 
    if config["pretrained_backbone"]:

        # The directory and file name for the loading the pretrained base model.
        res_directory = directory + config["output_path"] + "models/"
        file_name = args.dataset + "-" + args.model + "-" + str(config["num_ways"]) + "way.pth"

        # Loading the base model from the .pth file
        base_model_loaded = torch.load(res_directory + file_name, map_location=torch.device('cpu'))

        # Creating a base model instances and loading in the state dictionary.
        base_model = model(**config).to(device)
        base_model.load_state_dict(base_model_loaded)
    
    else:  # If we are using an untrained backbone.
        base_model = model(**config).to(device)

    # Creating the base model's *meta* optimizer.
    meta_optimizer = optimizer_archive[config["meta_optimizer_name"]](
        base_model.parameters(), **config["meta_optimizer_settings"])

    # Creating the base model's *base* optimizer.
    base_optimizer = optimizer_archive[config["base_optimizer_name"]](
        base_model.parameters(), **config["base_optimizer_settings"])

    # Creating the *function* for the bootstrapped optimizer.
    base_bootstrapped_optimizer = functools.partial(
        optimizer_archive[config["base_bootstrapped_optimizer_name"]],
        **config["base_bootstrapped_optimizer_settings"])

    # Creating the meta learning rate scheduler.
    meta_scheduler = scheduler_archive[config["meta_scheduler_name"]](
        meta_optimizer, **config["meta_scheduler_settings"])

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "maml-" + args.dataset + "-" + args.model + "-" + \
                str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot-" + str(random_state)

    print("maml", args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    # Performing the meta-training phase.
    meta_training_history, base_model, _ = meta_training(
        base_model, meta_optimizer, base_optimizer, base_bootstrapped_optimizer,
        meta_scheduler, training, validation,
        matching_loss_function=objective_archive[config["matching_loss_fn"]],
        meta_loss_function=objective_archive[config["meta_loss_fn"]],
        base_loss_function=objective_archive[config["base_loss_fn"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        **config
    )

    # Recording the end of the meta-training phase.
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Exporting the learned model's state dictionary.
    export_model(base_model, res_directory, file_name)

    # Performing the meta-testing phase.
    results["training_mean"], results["training_ci"] = meta_testing(
        base_model, base_optimizer, training,
        loss_function=objective_archive[config["base_loss_fn"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        **config
    )
    results["testing_mean"], results["testing_ci"] = meta_testing(
        base_model, base_optimizer, testing,
        loss_function=objective_archive[config["base_loss_fn"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        **config
    )

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Recording the training history.
    results["meta_training_history"] = meta_training_history

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)

    print("maml", args.dataset, args.model, "seed", str(random_state), "complete")


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
