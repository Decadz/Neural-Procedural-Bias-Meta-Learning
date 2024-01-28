import collections
import torch


class AdaLossNetwork(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, task_embedding_size, reduction="mean", **kwargs):
        super(AdaLossNetwork, self).__init__()

        # Number of classes used in the current problem.
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots
        self.reduction = reduction

        # Embedding size for making loss function task adaptive.
        self.task_embedding_size = task_embedding_size

        # Defining the inductive loss networks architecture.
        self.inductive_network = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMLinearBlock(num_ways * 2 + 1, num_ways * 2 + 1, torch.nn.ReLU, task_embedding_size)),
            ("block2", _FiLMLinearBlock(num_ways * 2 + 1, num_ways * 2 + 1, torch.nn.ReLU, task_embedding_size)),
            ("block3", _FiLMLinearBlock(num_ways * 2 + 1, 1, SmoothLeakyRelU, task_embedding_size))
        ]))

        # Defining the transductive loss networks architecture.
        self.transductive_network = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMLinearBlock(num_ways * 2 + 1, num_ways * 2 + 1, torch.nn.ReLU, task_embedding_size)),
            ("block2", _FiLMLinearBlock(num_ways * 2 + 1, num_ways * 2 + 1, torch.nn.ReLU, task_embedding_size)),
            ("block3", _FiLMLinearBlock(num_ways * 2 + 1, 1, SmoothLeakyRelU, task_embedding_size))
        ]))

        # Initializing the loss networks parameters.
        self.initialize()
        
    def forward(self, fx, z, y):

        # Calculating the inductive learned loss value.
        inductive_loss = self._calculate_inductive_loss(fx, y)

        # Calculating the transductive learned loss value.
        transductive_loss = self._calculate_transductive_loss(fx, z)

        # Calculating the task loss (which helps add bias for initialization).
        task_loss = self._calculate_task_loss(fx, y)

        # Returning the learned loss + task loss value.
        return inductive_loss + transductive_loss + task_loss

    def _calculate_inductive_loss(self, fx, y):

        # Partitioning out the support predictions from f(x).
        fx_support, _ = torch.split(fx, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)
        #fx_support = fx

        # One-hot encoding the support target.
        y_support = torch.nn.functional.one_hot(y, num_classes=self.num_ways)

        # Computing the cross entropy which is given as an input to the inductive loss.
        cross_entropy = torch.nn.functional.cross_entropy(fx_support, y, reduction="none").unsqueeze(1)

        # Computing the learned loss for each instance.
        learned_inductive_loss = self.inductive_network(torch.cat((fx_support, y_support, cross_entropy), dim=1))

        # Reducing the vector of learned loss values into a scalar.
        return self._reduce_output(learned_inductive_loss)

    def _calculate_transductive_loss(self, fx, z):

        # Partitioning out the support and query embeddings from z, and query predictions from f(x).
        z_support, z_query = torch.split(z, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)
        _, fx_query = torch.split(fx, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)

        # Computing the prototypes for each of the ways (classes).
        prototypes = z_support.reshape(self.num_shots, self.num_ways, -1).mean(dim=0)

        # Computing each query instances euclidean distance to each of the prototypes.
        query = z_query.unsqueeze(1).expand(z_query.shape[0], prototypes.shape[0], -1)
        prototypes = prototypes.unsqueeze(0).expand(query.shape[0], prototypes.shape[0], -1)
        y_approx = - ((query - prototypes) ** 2).sum(dim=2)

        # Computing the cross entropy which is given as an input to the inductive loss.
        cross_entropy = torch.nn.functional.cross_entropy(fx_query, y_approx, reduction="none").unsqueeze(1)

        # Computing the learned loss for each instance.
        learned_transductive_loss = self.inductive_network(torch.cat((fx_query, y_approx, cross_entropy), dim=1))

        # Reducing the vector of learned loss values into a scalar.
        return self._reduce_output(learned_transductive_loss)

    def _calculate_task_loss(self, fx, y):

        # Partitioning out the support and query predictions from f(x).
        fx_support, _ = torch.split(fx, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)
        #fx_support = fx

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

    def adapt_representation(self, task_embedding):
        # Setting the FiLM adapt settings for all inductive loss network layers.
        for name, module in self.inductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.adapt_representation(task_embedding)

        # Setting the FiLM adapt settings for all transductive loss network layers.
        for name, module in self.transductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.adapt_representation(task_embedding)

    def reset_gamma_beta(self):
        # Resetting the gamma and beta values in the FiLM layers.
        for name, module in self.inductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.reset_gamma_beta()

        # Resetting the gamma and beta values in the FiLM layers.
        for name, module in self.transductive_network.named_children():
            if isinstance(module, _FiLMLinearBlock):
                module.reset_gamma_beta()

    def meta_parameters(self):
        for module in self.inductive_network.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        for module in self.transductive_network.children():
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

    def __init__(self, in_features, out_features, activation, task_embedding_size):
        super(_FiLMLinearBlock, self).__init__()

        # The underlying linear layer (as implemented in PyTorch).
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(task_embedding_size, out_features * 2)

        # The non-linear activation function.
        self.activation = activation()

        # The multiplicative (gamma) and additive (beta) linear transformation values.
        self.register_buffer("gamma", torch.zeros(out_features, requires_grad=False))
        self.register_buffer("beta", torch.zeros(out_features, requires_grad=False))

        # The size of the task embedding.
        self.task_embedding_size = task_embedding_size

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.linear(x)

        # Expanding tensor back into the correct dimension size.
        gamma = self.gamma[None, :].expand_as(z)
        beta = self.beta[None, :].expand_as(z)

        return self.activation((1 + gamma) * z + beta)

    def initialize(self):
        torch.nn.init.normal_(self.linear.weight, 0, 0.01)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)

    def adapt_representation(self, task_embedding):
        # Computing the gamma and beta weights using the given task embedding.
        self.gamma, self.beta = self.film(task_embedding).chunk(2)

    def reset_gamma_beta(self):
        self.gamma = torch.zeros(self.task_embedding_size, requires_grad=False)
        self.beta = torch.zeros(self.task_embedding_size, requires_grad=False)

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
