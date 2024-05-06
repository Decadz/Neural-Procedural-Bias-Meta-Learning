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

# python experiments/run_meta_testing.py --dataset miniimagenet --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 100 --device cuda:0

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

    # Loading the base model from the .pth file.
    base_model_state_dictionary = torch.load(res_directory + "models/" + file_name, map_location=torch.device('cpu'))

    # Creating a base model instances and loading in the state dictionary.
    base_model = model(**config).to(device)

    # Loading the loss function from the .pth file.
    learned_loss_state_dictionary = torch.load(res_directory + "losses/" + file_name, map_location=torch.device('cpu'))

    # Creating the meta learned loss function.
    learned_loss = AdaLossNetwork(model=base_model, **config).to(device)

    # Loading the base model's state dictionary from the .pth file
    task_encoder_state_dict = torch.load(
        "source/models/pretrained/" + args.dataset + "/" + args.dataset +
        "-relationnet-" + str(config["num_ways"]) + "way.pth",
        map_location=torch.device('cpu')
    )

    # Creating a base model instances and loading in the state dictionary.
    task_encoder = RelationNetwork(**config).to(device)
    task_encoder.load_state_dict(task_encoder_state_dict)

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "npbml-testing-" + args.dataset + "-" + args.model + "-" + \
                str(config["num_ways"]) + "way-" + str(config["num_shots"]) + "shot-" + str(random_state)

    # Creating a results dictionary and recording the start time of the experiment.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    print("meta-testing", args.dataset, args.model, "seed", str(random_state), "started")

    # Performing the meta-testing phase.
    training_mean, training_ci = _meta_testing(
        base_model=base_model, base_model_state_dictionary=base_model_state_dictionary,
        loss_function=learned_loss, loss_function_state_dictionary=learned_loss_state_dictionary,
        dataset=training, task_encoder=task_encoder, config=config,
        performance_metric=objective_archive[config["evaluation_metric"]]
    )
    testing_mean, testing_ci = _meta_testing(
        base_model=base_model, base_model_state_dictionary=base_model_state_dictionary,
        loss_function=learned_loss, loss_function_state_dictionary=learned_loss_state_dictionary,
        dataset=testing,  task_encoder=task_encoder, config=config,
        performance_metric=objective_archive[config["evaluation_metric"]]
    )

    # Recording the end of the meta-training phase.
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Recording the training and testing performance.
    results["training_mean"], results["training_ci"] = training_mean, training_ci
    results["testing_mean"], results["testing_ci"] = testing_mean, testing_ci
    print("Training Performance:", results["training_mean"])
    print("Testing Performance:", results["testing_mean"])

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Recording information about the experiment.
    results["command"] = "python " + " ".join(sys.argv)  # Recording the python command used.

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)
    print("meta-testing", args.dataset, args.model, "seed", str(random_state), "complete")


def _meta_testing(base_model, base_model_state_dictionary, loss_function, loss_function_state_dictionary,
                  dataset, task_encoder, performance_metric, config):

    # Setting the base model to inference mode.
    base_model.eval()

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

        # Sampling a batch of support and query instances.
        X_support, y_support, X_query, y_query = next(dataset)

        # Merging the support and query into one batch.
        X_support_query = torch.cat((X_support, X_query), dim=0)

        # Resetting the classification head to ensure permutation invariance.
        base_model.reset_classifier()

        # Generating the global task embedding and relation scores.
        with torch.no_grad():
            task_embeddings = task_encoder(X_support_query)

        # Taking a predetermined number of inner steps before meta update.
        for inner_step in range(config["base_gradient_steps"]):

            # Clearing out the gradient cache.
            base_optimizer.zero_grad()

            # Computing the predictions on support set and computing the loss.
            fx = base_model(X_support_query, task_adaptive=True)
            loss_support = loss_function(fx, y_support, task_embeddings, base_model)

            # Updating the model weights.
            loss_support.backward()
            base_optimizer.step()

        # Computing the base network predictions on query set.
        with torch.no_grad():
            yp_query = base_model(X_query, task_adaptive=True)

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
