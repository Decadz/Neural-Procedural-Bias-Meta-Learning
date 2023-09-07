import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *
from source import *

import functools
import torch


# TODO - FiLM for convolutions
# TODO - Function for resetting batch normalization statistics.

"""
def init_adaptation(self):
        # Reset BN running stats
        for m in self.modules():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()
"""


class Parameterization(torch.nn.Module):

    def __init__(self, fan_in, fan_out, rank=4, lora=True, warp=True, activation=torch.nn.ReLU):

        """
        
        :param fan_in: The number of inputs to the layer.
        :param fan_out: The number of outputs to the layer.
        :param rank: The rank of the learned decomposition.
        :param lora: Whether LoRA weights are used in the parameterization.
        :param warp: Whether Warp weights are used in the parameterization.
        :param activation: Activation to use in the warp layer.
        """
        
        super().__init__()

        # Generating the LoRA decomposition parameters.
        if lora:
            self.lora_A = torch.nn.Parameter(torch.zeros((fan_in, rank)))
            self.lora_B = torch.nn.Parameter(torch.zeros((rank, fan_out)))
            torch.nn.init.kaiming_uniform_(self.lora_A)

        # Generating the Warp decomposition parameters.
        if warp:
            self.warp_A = torch.nn.Parameter(torch.zeros((fan_out, rank)))
            self.warp_B = torch.nn.Parameter(torch.zeros((rank, fan_out)))
            torch.nn.init.kaiming_uniform_(self.warp_A)
            torch.nn.init.kaiming_uniform_(self.warp_B)

        # Recording the parameterization settings.
        self.rank, self.lora, self.warp = rank, lora, warp
        self.activation = activation()

        # The forward function of the parameterization.
        self.forward_fn = self.npbml_forward

    def npbml_forward(self, X):
        if self.lora and self.warp:  # If we are using LoRA and Warp layers.
            return X + self.activation(self.lora_A.matmul(self.lora_B).matmul(
                self.warp_A).matmul(self.warp_B)).T
        
        elif self.lora:  # If we are only using LoRA layers.
            return X + self.lora_A.matmul(self.lora_B).T

        elif self.warp:  # If we are only using Warp layers.
            return X + self.activation(self.warp_A.matmul(self.warp_B)).T

        return X # If we are using neither return identity.

    def forward(self, X):
        # The forward pass to use given the current object state
        return self.forward_fn(X)

    def disable_npbml(self):
        self.forward_fn = lambda x: x

    def enable_npbml(self):
        self.forward_fn = self.npbml_forward

    @classmethod
    def from_linear(cls, layer, rank=4, lora=True, warp=True):
        """Factory for generating the parameterization for linear layers."""
        fan_out, fan_in = layer.weight.shape
        return cls(fan_in, fan_out, rank=rank, lora=lora, warp=warp)

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora=True, warp=True):
        """Factory for generating the parameterization for convolutional layers."""
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(fan_in, fan_out, rank=rank, lora=lora, warp=warp)


# Specify which layers to add the parameterization to, by default only adding to linear layers.
parameterization_config = { 
    torch.nn.Linear: {
        "weight": functools.partial(Parameterization.from_linear, rank=1, lora=True, warp=True),
    },
}


# ------------------- Helper functions for applying and removing parameterization -------------------

def apply_npbml(layer, register=True, config=parameterization_config):
    """add npbml parametrization to a layer, designed to be used with model.apply"""
    if register:
        if type(layer) in config:
            for attr_name, parametrization in config[type(layer)].items():
                torch.nn.utils.parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution.
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                torch.nn.utils.parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=False)


def add_npbml(model, config=parameterization_config):
    """add npbml parametrization to all layers in a model. Calling it twice will add npbml twice"""
    model.apply(functools.partial(apply_npbml, config=config))


def add_npbml_by_name(model, target_module_names, config=parameterization_config):
    """Add npbml parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_npbml(layer, config=config)

def remove_npbml(model):
    """remove npbml parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(functools.partial(apply_npbml, register=False))

def remove_npbml_by_name():
    pass  # TODO - Implement.

def enable_npbml(model):
    model.apply(apply_to_npbml(lambda x: x.enable_npbml()))

def disable_npbml(model):
    model.apply(apply_to_npbml(lambda x: x.disable_npbml()))


# ------------------- Helper functions for collecting parameters for training/saving -------------------

def _get_params_by_name(model, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            yield p

def lora_parameters(model):
    return _get_params_by_name(model, name_filter=_name_is_lora)

def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if _name_is_lora(k)}

def _name_is_lora(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )

def warp_parameters(model):
    return _get_params_by_name(model, name_filter=_name_is_warp)

def warp_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if _name_is_warp(k)}

def _name_is_warp(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["warp_A", "warp_B"]
    )

# ------------------- Helper functions for interacting with multiple npbml -------------------

# TODO - These still need to be updated.

def apply_to_npbml(fn):
    # apply a function to Parameterization layers, designed to be used with model.apply
    def apply_fn(layer):
        if isinstance(layer, Parameterization):
            fn(layer)

    return apply_fn

def load_multiple_npbml(model, npbml_state_dicts):
    model.apply(apply_to_npbml(_prepare_for_multiple_npbml))
    for state_dict in npbml_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_npbml(_append_npbml))
    return model

def select_npbml(model, index):
    model.apply(apply_to_npbml(lambda x: _select_npbml(x, index)))
    return model

def _prepare_for_multiple_npbml(npbml_layer):
    npbml_layer.lora_As = []
    npbml_layer.lora_Bs = []

def _append_npbml(npbml_layer):
    npbml_layer.lora_As.append(torch.nn.Parameter(npbml_layer.lora_A.clone()))
    npbml_layer.lora_Bs.append(torch.nn.Parameter(npbml_layer.lora_B.clone()))

def _select_npbml(npbml_layer, index):
    npbml_layer.lora_A = npbml_layer.lora_As[index]
    npbml_layer.lora_B = npbml_layer.lora_Bs[index]




torch.cuda.manual_seed_all(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)


# python3 source/parameterization.py
 
print("\n\n")

# A simple demonstration model.
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=3, bias=False),
)

x = torch.randn(1, 2)

# Example forward pass using base model.
y = model(x)
print("y1", y)

print("\n====================================\n")

# To make the output different, we need to initialize B to something non-zero
add_npbml(model)
model.apply(apply_to_npbml(lambda x: torch.nn.init.ones_(x.lora_B)))
y_new = model(x)
print("y2", y_new)

print("\n====================================\n")

print("\nx", x)
print("\nmodel", list(model.parameters()))
print("\nlora", list(lora_parameters(model)))
print("\nwarp", list(warp_parameters(model)))

print("\n====================================\n")