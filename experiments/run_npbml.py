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

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting the experiments directory for loading and saving.
directory = os.path.dirname(os.path.abspath(__file__)) + "/"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Parsing arguments to construct experiments.
# ============================================================

# Reading in all the experimental configurations and settings.
parser = argparse.ArgumentParser(description="Experiment Runner")
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
        pretrained_directory = "source/models/pretrained/" + args.dataset + "/"
        file_name = args.dataset + "-" + args.model + "-" + str(config["num_ways"]) + "way.pth"

        # Loading the base model's state dictionary from the .pth file
        base_model_state_dict = torch.load(pretrained_directory + file_name, map_location=torch.device('cpu'))

        # Creating a base model instances and loading in the state dictionary.
        base_model = model(**config).to(device)
        base_model.load_state_dict(base_model_state_dict)

        # Disable gradient tracking for the specified layers
        for layer in [base_model.encoder.block1, base_model.encoder.block2, base_model.encoder.block3]:
            for param in layer.parameters():
                param.requires_grad = False

    else:  # If we are using an untrained backbone.
        base_model = model(**config).to(device)

    # Loading the base model's state dictionary from the .pth file
    task_encoder_state_dict = torch.load(
        "source/models/pretrained/" + args.dataset + "/" + args.dataset +
        "-relationnet-" + str(config["num_ways"]) + "way.pth",
        map_location=torch.device('cpu')
    )

    # Creating a base model instances and loading in the state dictionary.
    task_encoder = RelationNetwork(**config).to(device)
    task_encoder.load_state_dict(task_encoder_state_dict)

    # Creating the meta learned loss function.
    learned_loss = AdaLossNetwork(model=base_model, **config).to(device)

    meta_optimizer = optimizer_archive[config["meta_optimizer_name"]](
       list(base_model.meta_parameters()) + list(learned_loss.meta_parameters()),
       **config["meta_optimizer_settings"])

    # Creating the base model's *base* optimizer.
    base_optimizer = optimizer_archive[config["base_optimizer_name"]](
        base_model.base_parameters(), **config["base_optimizer_settings"])

    # Creating the meta learning rate scheduler.
    meta_scheduler = scheduler_archive[config["meta_scheduler_name"]](
        meta_optimizer, **config["meta_scheduler_settings"])

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "npbml-" + args.dataset + "-" + args.model + "-" + \
                str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot-" + str(random_state)

    # Creating a results dictionary and recording the start time of the experiment.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    print("npbml", args.dataset, args.model, "seed", str(random_state), "started")

    # Performing the meta-training phase.
    meta_training_history, base_model, learned_loss = meta_training_npbml(
        base_model, meta_optimizer, base_optimizer, meta_scheduler, training, validation,
        task_encoder=task_encoder, base_loss_function=learned_loss,
        meta_loss_function=objective_archive[config["meta_loss_fn"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        **config
    )

    # Recording the end of the meta-training phase.
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Exporting the learned model's state dictionary.
    export_model(base_model, res_directory, file_name)
    export_loss(learned_loss, res_directory, file_name)

    # Performing the meta-testing phase.
    results["training_mean"], results["training_ci"] = meta_testing_npbml(
        base_model, base_optimizer, training, task_encoder=task_encoder, loss_function=learned_loss,
        performance_metric=objective_archive[config["evaluation_metric"]], **config
    )
    results["testing_mean"], results["testing_ci"] = meta_testing_npbml(
        base_model, base_optimizer, testing, task_encoder=task_encoder, loss_function=learned_loss,
        performance_metric=objective_archive[config["evaluation_metric"]], **config
    )

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Recording the training history.
    results["meta_training_history"] = meta_training_history

    # Recording information about the experiment.
    results["command"] = "python " + " ".join(sys.argv)  # Recording the python command used.

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)

    print("npbml", args.dataset, args.model, "seed", str(random_state), "complete")


# Loading the relevant methods configurations file.
dataset_config = yaml.safe_load(open(dataset_config_archive[args.dataset]))
method_config = yaml.safe_load(open(method_config_archive["npbml"]))

# Generating the final experimental configurations.
required_args = {"dataset", "model", "seeds", "device"}
config = override_configurations(args, args_unknown, required_args, dataset_config, method_config)

# Retrieving the function for the selected dataset.
dataset_fn = dataset_archive[args.dataset]

# Retrieving the function for the selected model.
model_fn = model_archive[args.model]

# Executing the experiments with the given arguments.
for random_state in args.seeds:
    _run_experiment(dataset_fn, model_fn, config, random_state)
