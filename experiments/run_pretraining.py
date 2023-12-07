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
    training, validation, _ = dataset(pretraining=True, device=device, **config)

    # Creating the base model.
    base_model = model(
        input_channels=config["input_channels"],
        num_ways=training.dataset.num_classes
    ).to(device)

    # If using a Conv4 base model train using Adam.
    if "conv" in args.model:
        pretrain_optimizer = torch.optim.Adam(
            base_model.adapt_parameters(), lr=0.001, weight_decay=0.0005)

        pretrain_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            pretrain_optimizer, milestones=[40000, 60000, 80000, 100000], gamma=0.1)

    # If using a ResNet base model train using SGD.
    elif "resnet" in args.model:
        pretrain_optimizer = torch.optim.SGD(
            base_model.adapt_parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0005)

        pretrain_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            pretrain_optimizer, milestones=[80000, 100000, 110000, 120000], gamma=0.1)

    # Creating a results dictionary and recording the start time of the experiment.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    print("pretraining", args.dataset, args.model, "seed", str(random_state), "started")

    base_model, meta_history, fine_tuning_history = pretrain(
        base_model, pretrain_optimizer, pretrain_scheduler, training, validation,
        gradient_steps=config["pretraining_gradient_steps"],
        batch_size=config["pretraining_batch_size"],
        loss_function=objective_archive[config["base_loss_fn"]],
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
    pretrained_directory = "source/models/pretrained/" + args.dataset + "/"
    export_model(base_model, pretrained_directory, args.dataset + "-" + args.model +
                 "-" + str(config["num_ways"]) + "way", separate_models_directory=False)

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Recording the training history.
    results["meta_history"] = meta_history
    results["fine_tuning_history"] = fine_tuning_history

    # Exporting the results to a json file.
    res_directory = directory + config["output_path"]
    file_name = "pretraining-" + args.dataset + "-" + args.model + "-" + str(config["num_ways"]) + "way"

    # Recording information about the experiment.
    results["command"] = "python " + " ".join(sys.argv)  # Recording the python command used.

    export_results(results, res_directory, file_name)

    print("pretraining", args.dataset, args.model, "seed", str(random_state), "complete")


# Loading the relevant methods configurations file.
dataset_config = yaml.safe_load(open(dataset_config_archive[args.dataset]))
method_config = yaml.safe_load(open(method_config_archive["pretraining"]))

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
