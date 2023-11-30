import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *
from source import *

import argparse
import torch
import random
import numpy
import yaml
import json

# python experiments/run_testing.py --method maml --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 1 --base_gradient_steps 5 --seeds 1008

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
parser.add_argument("--method", required=True, type=str)
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
    file_name = args.method + "-" + args.dataset + "-" + args.model + "-" + \
                str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot-" + str(random_state)
    
    # Loading the base model from the .pth file
    base_model_loaded = torch.load(res_directory + "models/" + file_name + ".pth",
                                   map_location=torch.device('cpu'))

    # If the state dictionary was saved load into the base model.
    if isinstance(base_model_loaded, dict):
        print("loaded state dictionary")
        base_model = model(**config).to(device)
        base_model.load_state_dict(base_model_loaded)

    # Else the whole model was saved, so overload base model object.
    else:  
        print("loaded full model")
        base_model = base_model_loaded

    # Creating the base model's *base* optimizer.
    base_optimizer = optimizer_archive[config["base_optimizer_name"]](
        base_model.parameters(), **config["base_optimizer_settings"])

    print(args.method, args.dataset, args.model, "seed", str(random_state), "started")

    # Loading the results ".json" file (dictionary) into memory.
    with open(res_directory + file_name + ".json", "r") as json_file:
        results = json.load(json_file)

    # Performing the meta-testing phase.
    training_mean, training_std = meta_testing(
        base_model, base_optimizer, training,
        loss_function=objective_archive[config["base_loss_fn"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        **config
    )
    testing_mean, testing_std = meta_testing(
        base_model, base_optimizer, testing,
        loss_function=objective_archive[config["base_loss_fn"]],
        performance_metric=objective_archive[config["evaluation_metric"]],
        **config
    )

    print("training_mean", training_mean)
    print("training_std", training_std)
    print("testing_mean", testing_mean)
    print("testing_std", testing_std)

    # Exporting the results to a json file.
    #export_results(results, res_directory, file_name)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "complete")


# Loading the relevant methods configurations file.
dataset_config = yaml.safe_load(open(dataset_config_archive[args.dataset]))
method_config = yaml.safe_load(open(method_config_archive[args.method]))

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
