import collections
import torch


class AdaLossNetwork(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, model, reduction="mean", **kwargs):
        super(AdaLossNetwork, self).__init__()

        # Number of classes used in the current problem.
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots
        self.reduction = reduction

        # Defining the inductive loss networks architecture.
        self.inductive_network = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMLinearBlock(num_ways * 2 + 1, 40, torch.nn.ReLU)),
            ("block2", _FiLMLinearBlock(40, 40, torch.nn.ReLU)),
            ("block3", _FiLMLinearBlock(40, 1, SmoothLeakyRelU))
        ]))

        # Defining the transductive loss networks architecture.
        self.transductive_network = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMLinearBlock(num_ways * 2 + 1, 40, torch.nn.ReLU)),
            ("block2", _FiLMLinearBlock(40, 40, torch.nn.ReLU)),
            ("block3", _FiLMLinearBlock(40, 1, SmoothLeakyRelU))
        ]))

        # Defining the regularization penalty networks architecture.
        layer_width = len(list(model.base_parameters())) * 4
        self.regularization_network = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMLinearBlock(layer_width, 40, torch.nn.ReLU)),
            ("block2", _FiLMLinearBlock(40, 40, torch.nn.ReLU)),
            ("block3", _FiLMLinearBlock(40, 1, SmoothLeakyRelU))
        ]))

        # Initializing the loss networks parameters.
        self.initialize()

    def forward(self, fx, y, task_embeddings, model):

        # Calculating the inductive learned loss value.
        inductive_loss = self._calculate_inductive_loss(fx, y)

        # Calculating the transductive learned loss value.
        transductive_loss = self._calculate_transductive_loss(fx, task_embeddings)

        # Calculating the learned regularization penalty.
        regularization_penalty = self._calculate_regularization_penalty(model)

        # Calculating the task loss (which helps add bias for initialization).
        task_loss = self._calculate_task_loss(fx, y)

        # Returning the learned loss + task loss value.
        return task_loss + inductive_loss + transductive_loss + regularization_penalty

    def _calculate_inductive_loss(self, fx, y):

        # Partitioning out the support predictions from f(x).
        fx_support, _ = torch.split(fx, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)

        # One-hot encoding the support target.
        y_support = torch.nn.functional.one_hot(y, num_classes=self.num_ways)

        # Computing the cross entropy which is given as an input to the inductive loss.
        cross_entropy = torch.nn.functional.cross_entropy(fx_support, y, reduction="none").unsqueeze(1)

        # Computing the learned loss for each instance.
        learned_inductive_loss = self.inductive_network(torch.cat((fx_support, y_support, cross_entropy), dim=1))

        # Reducing the vector of learned loss values into a scalar.
        return self._reduce_output(learned_inductive_loss)

    def _calculate_transductive_loss(self, fx, task_embeddings):

        # Partitioning out the support and query embeddings from z, and query predictions from f(x).
        _, fx_query = torch.split(fx, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)

        # Computing the cross entropy which is given as an input to the inductive loss.
        relation_scores_int = torch.argmax(task_embeddings, dim=1)
        cross_entropy = torch.nn.functional.cross_entropy(fx_query, relation_scores_int, reduction="none").unsqueeze(1)

        # Computing the learned loss for each instance.
        learned_transductive_loss = self.inductive_network(torch.cat((fx_query, task_embeddings, cross_entropy), dim=1))

        # Reducing the vector of learned loss values into a scalar.
        return self._reduce_output(learned_transductive_loss)

    def _calculate_regularization_penalty(self, model):

        # List for keeping track of all the layer statistics.
        regularization_network_x = []

        # Iterating over all the layers that are used for adaptation.
        for param in model.base_parameters():

            # Computing the mean, standard deviation, L1 norm, and L2 norm.
            mu, std = torch.mean(param.detach()), torch.std(param.detach())
            l1, l2 = torch.norm(param, p=2), torch.norm(param, p=2)

            # Stacking them into one tensor and recording it in the input list.
            regularization_network_x.append(torch.stack((mu, std, l1, l2)))

        # Combining all the layer statistics into one long vector.
        x = torch.cat(regularization_network_x)

        # Returning the learned regularization penalty.
        return self.regularization_network(x.unsqueeze(dim=0))

    def _calculate_task_loss(self, fx, y):

        # Partitioning out the support and query predictions from f(x).
        fx_support, _ = torch.split(fx, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)

        # Computing the task loss value.
        return torch.nn.functional.cross_entropy(fx_support, y)

    def initialize(self):
        # Initializing the inductive networks parameters.
        for name, module in self.inductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.initialize()

        # Initializing the transductive networks parameters.
        for name, module in self.transductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.initialize()

        # Initializing the regularization networks parameters.
        for name, module in self.regularization_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.initialize()

    def meta_parameters(self):
        for module in self.inductive_network.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        for module in self.transductive_network.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        for module in self.regularization_network.children():
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
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_features, out_features * 2)

        # The non-linear activation function.
        self.activation = activation()

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.linear(x)

        # Computing the gamma and beta values.
        gamma, beta = self.film(x).chunk(2, dim=1)

        # Applying the scale and shift FiLM to the pre-activation output.
        z = (1 + gamma) * z + beta

        return self.activation(z)

    def initialize(self):
        torch.nn.init.normal_(self.linear.weight, 0, 0.01)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)

    def meta_parameters(self):
        yield from self.linear.parameters()
        yield from self.film.parameters()


class SmoothLeakyRelU(torch.nn.Module):

    def __init__(self, leak=0.25, smooth=5, **kwargs):
        super(SmoothLeakyRelU, self).__init__()
        self.leak = leak  # Leak hyper-parameter.
        self.smooth = smooth  # Smoothness hyper-parameter.

    def forward(self, x):
        # Don't call 'torch.log()' directly, else you will get numerical instability.
        return self.leak * x + (1 - self.leak) * torch.nn.functional.softplus(x, beta=self.smooth)
