import collections
import torch


class AdaLossNetwork(torch.nn.Module):

    def __init__(self, task_loss_fn, num_ways, reduction="mean", **kwargs):
        super(AdaLossNetwork, self).__init__()

        # Number of classes used in the current problem.
        self.task_loss_fn = task_loss_fn
        self.num_ways = num_ways
        self.reduction = reduction

        # Defining the loss functions architecture.
        self.inductive_network = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMLinearBlock(num_ways * 2, num_ways * 2, activation=torch.nn.ReLU)),
            ("block2", _FiLMLinearBlock(num_ways * 2, num_ways * 2, activation=torch.nn.ReLU)),
            ("block3", _FiLMLinearBlock(num_ways * 2, 1, activation=SmoothLeakyRelU))
        ]))

        # Initializing the loss networks parameters.
        self.initialize()
        """
        self.transductive_network = torch.nn.Sequential(
            _FiLMLinearBlock(in_features, in_features),
            _FiLMLinearBlock(in_features, in_features),
            _FiLMLinearBlock(in_features, 1)
        )
        """
    def forward(self, y_pred, y_target, adapt=False):

        # Setting the settings for all the FiLM layers.
        for name, module in self.inductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.adapt = adapt

        # Computing the task loss value.
        task_loss = self.task_loss_fn(y_pred, y_target)

        # One-Hot encoding the target.
        y_target = torch.nn.functional.one_hot(y_target, num_classes=self.num_ways)

        # Computing the learned loss for each instance.
        learned_loss = self.inductive_network(torch.cat((y_pred, y_target), dim=1))

        # Reducing the vector of learned loss values into a scalar.
        learned_loss = self._reduce_output(learned_loss)

        # Returning the learned loss + task loss value.
        return learned_loss + task_loss

    def initialize(self):
        # Initializing the networks parameters.
        for name, module in self.inductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.initialize()

    def meta_parameters(self):
        for module in self.inductive_network.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()

    def _reduce_output(self, loss):
        # Applying the desired reduction operation to the loss vector.
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class _FiLMLinearBlock(torch.nn.Module):

    def __init__(self, in_features, out_features, activation):
        super(_FiLMLinearBlock, self).__init__()

        # The underlying linear layer (as implemented in PyTorch).
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_features, out_features * 2)

        # The non-linear activation function.
        self.activation = activation()

        # The multiplicative (gamma) and additive (beta) linear transformation values.
        self.register_buffer("gamma", torch.zeros(out_features, requires_grad=False))
        self.register_buffer("beta", torch.zeros(out_features, requires_grad=False))

        # Field for controlling the current state of the layer.
        self.adapt = False

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.linear(x)

        # Computing feature wise linear modulation layer.
        if self.adapt is True:

            # Computing the gamma and beta weights and mean reducing in the batch dimension.
            self.gamma, self.beta = self.film(x).mean(dim=0).chunk(2)

        # Expanding tensor back into the correct dimension size.
        gamma = self.gamma[None, :].expand_as(z)
        beta = self.beta[None, :].expand_as(z)
        return self.activation((1 + gamma) * z + beta)

    def initialize(self):
        torch.nn.init.normal_(self.linear.weight, 0, 0.01)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)

    def meta_parameters(self):
        yield from self.linear.parameters()
        yield from self.film.parameters()


class SmoothLeakyRelU(torch.nn.Module):

    def __init__(self, leak=1, smooth=0.01, **kwargs):
        super(SmoothLeakyRelU, self).__init__()
        self.leak = leak  # Leak hyper-parameter.
        self.smooth = smooth  # Smoothness hyper-parameter.

    def forward(self, x):
        # Don't call 'torch.log()' directly, else you will get numerical instability.
        return self.leak * x + (1 - self.leak) * torch.nn.functional.softplus(x, beta=self.smooth)
