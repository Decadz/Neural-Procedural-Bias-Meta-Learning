# Importing the base-networks.
from experiments.resources.models.conv import Conv, WideConv
from experiments.resources.models.conv import LinearWarpConv, LinearWarpWideConv
from experiments.resources.models.conv import WarpConv, WarpWideConv
from experiments.resources.models.resnet import ResNet, WideResNet
from experiments.resources.models.resnet import LinearWarpResNet, LinearWarpWideResNet
from experiments.resources.models.resnet import WarpResNet, WarpWideResNet

# Importing the meta-learning datasets.
from experiments.resources.datasets import Omniglot
from experiments.resources.datasets import CIFARFS
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


dataset_archive = {
    "omniglot": {"data": Omniglot, "config": "experiments/resources/configurations/omniglot_config.yaml"},
    "fc100": {"data": FC100, "config": "experiments/resources/configurations/fc100_config.yaml"},
    "cifarfs": {"data": CIFARFS, "config": "experiments/resources/configurations/cifarfs_config.yaml"},
    "miniimagenet": {"data": MiniImagenet, "config": "experiments/resources/configurations/miniimagenet_config.yaml"},
    "tieredimagenet": {"data": TieredImagenet, "config": "experiments/resources/configurations/tieredimagenet_config.yaml"}
}


model_archive = {
    "conv": Conv,
    "wideconv": WideConv,
    "linearwarpconv": LinearWarpConv,
    "linearwarpwideconv": LinearWarpWideConv,
    "warpconv": WarpConv,
    "warpwideconv": WarpWideConv,
    "resnet": ResNet,
    "wideresnet": WideResNet,
    "linearwarpresnet": LinearWarpResNet,
    "linearwarpwideresnet": LinearWarpWideResNet,
    "warpresnet": WarpResNet,
    "warpwideresnet": WarpWideResNet,
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
