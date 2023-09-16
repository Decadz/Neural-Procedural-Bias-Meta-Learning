
# Importing the base-networks.
from experiments.resources.models.conv import Conv4, WarpConv4
from experiments.resources.models.resnet import ResNet12

# Importing the meta-learning datasets.
from experiments.resources.datasets import Omniglot
from experiments.resources.datasets import FC100
from experiments.resources.datasets import MiniImagenet
from experiments.resources.datasets import TieredImagenet

# Importing utility functions for running experiments.
from experiments.resources.parser import register_configurations
from experiments.resources.parser import override_configurations
from experiments.resources.metrics import MultiErrorRate
from experiments.resources.metrics import BinaryErrorRate
from experiments.resources.exporter import export_results
from experiments.resources.exporter import export_model

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

    # Turning the parameters into a flattened vector (tensor).
    p = torch.nn.utils.parameters_to_vector(p_target.parameters())
    q = torch.nn.utils.parameters_to_vector(q_target.parameters())

    # Converting into a probability distribution.
    p = torch.nn.functional.softmax(p, dim=0)
    q = torch.nn.functional.softmax(q, dim=0)

    # Computing the Kullback-Leibler divergence.
    return (p * (p / q).log()).sum()


dataset_archive = {
    "omniglot": {"data": Omniglot, "config": "experiments/resources/configurations/omniglot_config.yaml"},
    "fc100": {"data": FC100, "config": "experiments/resources/configurations/fc100_config.yaml"},
    "miniimagenet": {"data": MiniImagenet, "config": "experiments/resources/configurations/miniimagenet_config.yaml"},
    "tieredimagenet": {"data": TieredImagenet, "config": "experiments/resources/configurations/tieredimagenet.yaml"}
}


model_archive = {
    "conv4": Conv4,
    "warpconv4": WarpConv4,
    "resnet": ResNet12
}

objective_archive = {
    "multierrorrate": MultiErrorRate(),
    "binaryerrorrate": BinaryErrorRate(),
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
