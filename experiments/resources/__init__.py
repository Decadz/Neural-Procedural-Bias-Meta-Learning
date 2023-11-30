# Importing the meta-learning datasets.
from experiments.resources.datasets import Omniglot
from experiments.resources.datasets import CIFARFS
from experiments.resources.datasets import FC100
from experiments.resources.datasets import MiniImagenet
from experiments.resources.datasets import TieredImagenet

# Importing utility functions for running experiments.
from experiments.resources.parser import register_configurations
from experiments.resources.parser import override_configurations
from experiments.resources.metrics import ErrorRate
from experiments.resources.metrics import Accuracy
from experiments.resources.exporter import export_results
from experiments.resources.exporter import export_model
from experiments.resources.exporter import export_loss

import inspect
import torch


def match_signature(func):
    """ Matches the given arguments to the function signature. """
    def wrapped_func(*args, **kwargs):
        return func(*args, **{key: value for (key, value) in kwargs.items()
                              if key in inspect.signature(func).parameters})
    return wrapped_func


def kl_divergence(p_target, q_target):

    """
    Function for computing the Kullback-Leibler divergence given a target set 
    of parameters and a bootstrapped target to minimize the divergce to.
    """

    # Converting into a probability distribution.
    p = torch.nn.functional.softmax(p_target, dim=0)
    q = torch.nn.functional.softmax(q_target, dim=0)

    # Computing the Kullback-Leibler divergence.
    return (p * (p / q).log()).sum()


# Path to the configuration files for the datasets and methods.
dataset_config_path = "experiments/resources/configurations/datasets/"
method_config_path = "experiments/resources/configurations/methods/"

dataset_archive = {
    "omniglot": Omniglot,
    "fc100": FC100,
    "cifarfs": CIFARFS,
    "miniimagenet": MiniImagenet,
    "tieredimagenet": TieredImagenet
}

dataset_config_archive = {
    "omniglot": dataset_config_path + "omniglot_config.yaml",
    "fc100": dataset_config_path + "fc100_config.yaml",
    "cifarfs": dataset_config_path + "cifarfs_config.yaml",
    "miniimagenet": dataset_config_path + "miniimagenet_config.yaml",
    "tieredimagenet": dataset_config_path + "tieredimagenet_config.yaml",
}

method_config_archive = {
    "maml": method_config_path + "maml_config.yaml",
    "warpgrad": method_config_path + "warpgrad_config.yaml",
    "npbml": method_config_path + "npbml_config.yaml",
    "pretraining": method_config_path + "pretraining_config.yaml",
}

objective_archive = {
    "errorrate": ErrorRate(),
    "accuracy": Accuracy(),
    "nllloss": torch.nn.NLLLoss(),
    "bceloss": torch.nn.BCELoss(),
    "mseloss": torch.nn.MSELoss(),
    "celoss": torch.nn.CrossEntropyLoss(),
    "kldiv": kl_divergence
}

optimizer_archive = {
    "sgd": match_signature(torch.optim.SGD),
    "adam": match_signature(torch.optim.Adam)
}

scheduler_archive = {
    "multistep": match_signature(torch.optim.lr_scheduler.MultiStepLR),
    "exponential": match_signature(torch.optim.lr_scheduler.ExponentialLR),
    "cosine": match_signature(torch.optim.lr_scheduler.CosineAnnealingLR)
}
