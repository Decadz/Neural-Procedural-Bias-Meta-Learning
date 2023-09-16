import functools
import torch


class Parameterization(torch.nn.Module):

    def __init__(self, fan_in, fan_out, rank=4, lora=True, warp=True, activation=torch.nn.Identity):

        """
        A class for wrapping PyTorch modules with LoRA and Warp layers.
        
        :param fan_in: The number of inputs to the layer.
        :param fan_out: The number of outputs to the layer.
        :param rank: The rank of the learned decomposition.
        :param lora: Whether LoRA weights are used in the parameterization.
        :param warp: Whether Warp weights are used in the parameterization.
        :param activation: Activation to use in the warp layer.
        """
        
        super().__init__()

        # Recording the parameterization settings.
        self.rank, self.lora, self.warp = rank, lora, warp
        self.activation = activation()

        # Generating the LoRA decomposition parameters.
        if lora:
            self.lora_A = torch.nn.Parameter(torch.zeros((fan_in, rank)))
            self.lora_B = torch.nn.Parameter(torch.zeros((rank, fan_out)))
            torch.nn.init.xavier_uniform_(self.lora_A)
            torch.nn.init.zeros_(self.lora_B)

        # Generating the Warp decomposition parameters.
        if warp:
            self.warp_A = torch.nn.Parameter(torch.zeros((fan_out, rank)))
            self.warp_B = torch.nn.Parameter(torch.zeros((rank, fan_out)))
            torch.nn.init.xavier_uniform_(self.warp_A)
            torch.nn.init.xavier_uniform_(self.warp_B)

    def forward(self, X):
        if self.lora and self.warp:  # If we are using LoRA and Warp layers.
            return X + self.activation(self.lora_A.matmul(self.lora_B).matmul(
                self.warp_A).matmul(self.warp_B)).T.view(X.shape)
        
        elif self.lora:  # If we are only using LoRA layers.
            return X + self.lora_A.matmul(self.lora_B).T.view(X.shape)

        elif self.warp:  # If we are only using Warp layers.
            return X + self.activation(self.warp_A.matmul(self.warp_B)).T.view(X.shape)

        return X # If we are using neither return identity.

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


# ------------------- Configurations for different styles of parameterization -------------------


full_model_config = { 
    torch.nn.Linear: {
        "weight": functools.partial(Parameterization.from_linear, rank=4, lora=True, warp=True),
    },
    torch.nn.Conv2d: {
        "weight": functools.partial(Parameterization.from_conv2d, rank=4, lora=True, warp=True),
    },
}

head_only_config = { 
    torch.nn.Linear: {
        "weight": functools.partial(Parameterization.from_linear, rank=4, lora=True, warp=True),
    },
}

body_only_config = { 
    torch.nn.Conv2d: {
        "weight": functools.partial(Parameterization.from_conv2d, rank=4, lora=True, warp=True),
    },
}

# ------------------- Helper functions for applying and removing the parameterization -------------------


def apply_npbml(layer, register=True, config=full_model_config):
    """add npbml parametrization to a layer, designed to be used with model.apply"""
    if register:
        if type(layer) in config:
            for attr_name, parametrization in config[type(layer)].items():
                torch.nn.utils.parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution.
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                torch.nn.utils.parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=False)

def add_npbml(model, config=full_model_config):
    """add npbml parametrization to all layers in a model. Calling it twice will add npbml twice"""
    model.apply(functools.partial(apply_npbml, config=config))

def remove_npbml(model):
    """remove npbml parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(functools.partial(apply_npbml, register=False))

def add_npbml_by_name(model, target_module_names, config=full_model_config):
    """Add npbml parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_npbml(layer, config=config)

def remove_npbml_by_name(model, target_module_names):
    """Remove npbml parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            remove_npbml(layer)

def apply_to_npbml(fn):
    """Apply a function to Parameterization layers, designed to be used with model.apply"""
    def apply_fn(layer):
        if isinstance(layer, Parameterization):
            fn(layer)

    return apply_fn

# ------------------- Helper functions for gathering parameters for training/saving -------------------


def _get_params_by_name(model, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            yield p

def _name_is_original(name):
    return (
        not _name_is_lora(name) and
        not _name_is_warp(name)
    )

def _name_is_lora(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )

def _name_is_warp(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["warp_A", "warp_B"]
    )

def original_parameters(model):
    return _get_params_by_name(model, name_filter=_name_is_original)

def original_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if _name_is_original(k)}

def lora_parameters(model):
    return _get_params_by_name(model, name_filter=_name_is_lora)

def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if _name_is_lora(k)}

def warp_parameters(model):
    return _get_params_by_name(model, name_filter=_name_is_warp)

def warp_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if _name_is_warp(k)}
