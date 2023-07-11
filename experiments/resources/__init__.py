
# Importing the base-networks.
from experiments.resources.models.conv import Conv4a, Conv4b
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


dataset_archive = {
    "omniglot": {"data": Omniglot, "config": "experiments/resources/configurations/omniglot_config.yaml"},
    "fc100": {"data": FC100, "config": "experiments/resources/configurations/fc100_config.yaml"},
    "miniimagenet": {"data": MiniImagenet, "config": "experiments/resources/configurations/miniimagenet_config.yaml"},
    "tieredimagenet": {"data": TieredImagenet, "config": "experiments/resources/configurations/tieredimagenet.yaml"}
}


model_archive = {
    "conv4a": Conv4a,
    "conv4b": Conv4b,
    "resnet": ResNet12
}

objective_archive = {
    "multierrorrate": MultiErrorRate(),
    "binaryerrorrate": BinaryErrorRate(),
    "nllloss": torch.nn.NLLLoss(),
    "bceloss": torch.nn.BCELoss(),
    "mseloss": torch.nn.MSELoss(),
    "celoss": torch.nn.CrossEntropyLoss()
}

optimizer_archive = {
    "sgd": match_signature(torch.optim.SGD),
    "adam": match_signature(torch.optim.Adam)
}

scheduler_archive = {
    "multistep": match_signature(torch.optim.lr_scheduler.MultiStepLR),
    "exponential": match_signature(torch.optim.lr_scheduler.ExponentialLR),
    "cosineannealing": match_signature(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
}
