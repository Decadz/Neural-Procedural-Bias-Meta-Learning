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
import tqdm

# python experiments/run_npbml.py --dataset miniimagenet --model adaconv32 --num_ways 5 --num_shots 5 --meta_batch_size 2 --pretrained_backbone True --seeds 0 --device cuda:0

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

    res_directory = directory + config["output_path"]
    file_name = "npbml-" + args.dataset + "-" + args.model + "-" + str(config["num_ways"]) + \
                "way-" + str(config["num_shots"]) + "shot-" + str(random_state) + ".pth"

    # Loading the base model from the .pth file
    base_model_state_dictionary = torch.load(res_directory + "models/" + file_name, map_location=torch.device('cpu'))

    # Creating a base model instances and loading in the state dictionary.
    base_model = model(**config).to(device)

    learned_loss_state_dictionary = torch.load(res_directory + "losses/" + file_name, map_location=torch.device('cpu'))

    # Creating the meta learned loss function.
    learned_loss = AdaLossNetwork(
        num_ways=config["num_ways"],
        task_loss_fn=objective_archive[config["meta_loss_fn"]]
    ).to(device)

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "testing-" + args.dataset + "-" + args.model + "-" + \
                str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot-" + str(random_state)

    # Creating a results dictionary and recording the start time of the experiment.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    print("meta-testing", args.dataset, args.model, "seed", str(random_state), "started")

    # Performing the meta-testing phase.
    results["training_mean"], results["training_ci"] = _meta_testing(
        base_model=base_model, base_model_state_dictionary=base_model_state_dictionary,
        loss_function=learned_loss, loss_function_state_dictionary=learned_loss_state_dictionary,
        dataset=training, performance_metric=objective_archive[config["evaluation_metric"]], config=config
    )

    results["testing_mean"], results["testing_ci"] = _meta_testing(
        base_model=base_model, base_model_state_dictionary=base_model_state_dictionary,
        loss_function=learned_loss, loss_function_state_dictionary=learned_loss_state_dictionary,
        dataset=testing, performance_metric=objective_archive[config["evaluation_metric"]], config=config
    )

    # Recording the end of the meta-training phase.
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Recording information about the experiment.
    results["command"] = "python " + " ".join(sys.argv)  # Recording the python command used.

    # Exporting the results to a json file.
    #export_results(results, res_directory, file_name)
    print("Training Performance:", results["training_mean"])
    print("Testing Performance:", results["testing_mean"])

    print("meta-testing", args.dataset, args.model, "seed", str(random_state), "complete")


def _meta_testing(base_model, base_model_state_dictionary, loss_function, loss_function_state_dictionary,
                  dataset, performance_metric, config):

    # List for keeping track of the learning history.
    performance_history = []

    for _ in (tqdm.tqdm(range(config["test_tasks"]), position=1, dynamic_ncols=True, desc="Validating Performance",
                        disable=True if config["verbose"] <= 1 else False, leave=False)):

        # Loading the base model and learned loss function.
        base_model.load_state_dict(base_model_state_dictionary)
        loss_function.load_state_dict(loss_function_state_dictionary)

        # Creating the base model's *base* optimizer.
        base_optimizer = optimizer_archive[config["base_optimizer_name"]](
            base_model.base_parameters(), **config["base_optimizer_settings"])

        # base_optimizer = type(base_optimizer)(base_model.parameters(), **base_optimizer.defaults)

        # Sampling a batch of support and query instances.
        X_support, y_support, X_query, y_query = next(dataset)

        # Resetting the running statistics for all batch normalization layers.
        base_model.reset_batch_norm()

        # Taking a predetermined number of inner steps before meta update.
        for inner_step in range(config["base_gradient_steps"]):

            # Clearing out the gradient cache.
            base_optimizer.zero_grad()

            # Computing the predictions on support set and computing the loss.
            yp_support = base_model(X_support, inner_step == 0)
            loss_support = loss_function(yp_support, y_support, inner_step)

            # Updating the model weights.
            loss_support.backward(retain_graph=True)
            base_optimizer.step()

        # Computing the base network predictions on query set.
        yp_query = base_model(X_query).detach()

        # Storing the validation performance history.
        performance_history.append(performance_metric(yp_query, y_query).item())

    # Returning the mean and 95% confidence interval of the performance.
    performance = torch.tensor(performance_history)
    mean = torch.mean(performance).item()
    std = torch.std(performance).item()
    ci = 1.96 * (std / (len(performance) ** 0.5))
    return mean, ci


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
