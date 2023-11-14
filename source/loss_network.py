import torch
import copy

# ============================================================
# Reusable Helper Functions
# ============================================================


def _calculate_inductive_embedding(X_support, y_support, base_model, loss_fn, inductive_encoder,
                                   num_ways, num_shots, reduction=False):

    # Calculating the support predictions, f(x).
    yp_support = base_model(X_support)

    # Calculating the support losses, L(f(x), y).
    loss = loss_fn(yp_support, y_support)

    # Converting into its one-hot encoded form, y \in R^{num_ways}.
    y_support = torch.nn.functional.one_hot(y_support, num_classes=num_ways)

    if reduction is True:  # If we are performing instance-wise (mean) reduction.
        encoding = [inductive_encoder(torch.cat((yp, y, loss.unsqueeze(0))))
                    for yp, y, loss in zip(yp_support, y_support, loss)]
        return torch.stack(encoding, dim=0).mean(axis=0)

    else:  # If we are not performing instance-wise reduction.
        return inductive_encoder(torch.cat((yp_support.view(-1), y_support.view(-1), loss)))


def _calculate_transductive_embedding(X_support, y_support, X_query, y_query, base_model, loss_fn,
                                      transductive_encoder, num_ways, num_shots, reduction=False):

    # Computing the support and query embeddings using the base model.
    support = base_model.encoder(X_support).detach()
    query = base_model.encoder(X_query).detach()

    # Computing the prototypes for each of the ways (classes).
    prototypes = support.reshape(num_shots, num_ways, -1).mean(dim=0)

    # Computing each query instances euclidean distance to each of the prototypes.
    query = query.unsqueeze(1).expand(query.shape[0], prototypes.shape[0], -1)
    prototypes = prototypes.unsqueeze(0).expand(query.shape[0], prototypes.shape[0], -1)
    logits = - ((query - prototypes) ** 2).mean(dim=2)

    # Turning the distances into a probabilistic approximation of the ground truth.
    yp_query = torch.nn.functional.softmax(logits, dim=1)

    # Calculating the query losses, L(f(x), y).
    loss = loss_fn(yp_query, y_query)

    # Converting into its one-hot encoded form, y \in R^{num_ways}.
    y_query = torch.nn.functional.one_hot(y_query, num_classes=num_ways)

    if reduction is True:  # If we are performing instance-wise (mean) reduction.
        encoding = [transductive_encoder(torch.cat((yp, y, loss.unsqueeze(0))))
                    for yp, y, loss in zip(yp_query, y_query, loss)]
        return torch.stack(encoding, dim=0).mean(axis=0)

    else:  # If we are not performing instance-wise reduction.
        return transductive_encoder(torch.cat((yp_query.view(-1), y_query.view(-1), loss)))


def _calculate_task_embedding(base_model, task_encoder, reduction=False):
    x_task = []

    # Iterating over all the modules of the model.
    for module in base_model.modules():

        # If its a trainable linear or conv layer record the mean, std, L1 and L2 norm.
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            param = torch.cat([p.view(-1) for p in module.parameters()])
            x = torch.stack((torch.mean(param), torch.std(param), torch.norm(param, p=1), torch.norm(param, p=2)))
            x_task.append(x)

    if reduction is True:  # If we are performing instance-wise (mean) reduction.
        encoding = [task_encoder(x) for x in x_task]
        return torch.stack(encoding, dim=0).mean(axis=0)

    else:   # If we are not performing instance-wise reduction.
        return task_encoder(torch.cat(x_task))


def _calculate_num_layers(base_model):
    count = 0
    for module in base_model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            count += 1
    return count


def _standardize_inputs(self):
    return None


# ============================================================
# Loss Network Definitions
# ============================================================


class _SmoothLeakyReLU(torch.nn.Module):

    def __init__(self, beta, gamma, **kwargs):
        super(_SmoothLeakyReLU, self).__init__()
        self.gamma = gamma  # Leak hyper-parameter.
        self.beta = beta  # Smoothness hyper-parameter.

    def forward(self, x):
        # Don't call 'torch.log()' directly, else you will get numerical instability.
        return self.gamma * x + (1 - self.gamma) * torch.nn.functional.softplus(x, beta=self.beta)


class LearnedLossV1(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, meta_loss_fn, **kwargs):
        super(LearnedLossV1, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.meta_loss_fn = copy.deepcopy(meta_loss_fn)
        self.meta_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Inductive: {fx, y, L} = 2 * (num_ways * num_shots) + num_ways * (num_ways * num_shots)
        # Transductive: {fx, y^, L} = 2 * (num_ways * (num_ways * test_shots)) + num_ways * test_shots
        # Task: {mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2} = 4 * num_layers
        self.input_dim = 2 * (num_ways * (num_ways * num_shots)) + num_ways * num_shots + \
                         2 * (num_ways * (num_ways * test_shots)) + num_ways * test_shots + \
                         4 * _calculate_num_layers(base_model)

        # Defining the loss functions architecture.
        self.mixer_network = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )

    def forward(self, X_support, y_support, X_query, y_query, base_model):

        task_loss = self.meta_loss_fn(base_model(X_support), y_support).mean()

        # TODO - Inductive

        # Calculating the support predictions, f(x).
        yp_support = base_model(X_support)

        # Calculating the support losses, L(f(x), y).
        inductive_loss = self.meta_loss_fn(yp_support, y_support)

        # Converting into its one-hot encoded form, y \in R^{num_ways}.
        y_support = torch.nn.functional.one_hot(y_support, num_classes=self.num_ways)

        # TODO - Transductive

        # Computing the support and query embeddings using the base model.
        support = base_model.encoder(X_support).detach()
        query = base_model.encoder(X_query).detach()

        # Computing the prototypes for each of the ways (classes).
        prototypes = support.reshape(self.num_shots, self.num_ways, -1).mean(dim=0)

        # Computing each query instances euclidean distance to each of the prototypes.
        query = query.unsqueeze(1).expand(query.shape[0], prototypes.shape[0], -1)
        prototypes = prototypes.unsqueeze(0).expand(query.shape[0], prototypes.shape[0], -1)
        logits = - ((query - prototypes) ** 2).mean(dim=2)

        # Turning the distances into a probabilistic approximation of the ground truth.
        yp_query = torch.nn.functional.softmax(logits, dim=1)

        # Calculating the query losses, L(f(x), y).
        transductive_loss = self.meta_loss_fn(yp_query, y_query)

        # Converting into its one-hot encoded form, y \in R^{num_ways}.
        y_query = torch.nn.functional.one_hot(y_query, num_classes=self.num_ways)

        # TODO - Task

        x_task = []

        # Iterating over all the modules of the model.
        for module in base_model.modules():

            # If its a trainable linear or conv layer record the mean, std, L1 and L2 norm.
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                param = torch.cat([p.view(-1) for p in module.parameters()])
                x = torch.stack((torch.mean(param), torch.std(param), torch.norm(param, p=1), torch.norm(param, p=2)))
                x_task.append(x)

        # Concatenating all the information into a long vector.
        inputs = torch.cat((yp_support.view(-1), y_support.view(-1), inductive_loss,
                            yp_query.view(-1), y_query.view(-1), transductive_loss,
                            torch.cat(x_task)))

        learned_loss = self.mixer_network(inputs)
        return learned_loss + task_loss


class LearnedLossV2(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, meta_loss_fn, **kwargs):
        super(LearnedLossV2, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.meta_loss_fn = copy.deepcopy(meta_loss_fn)
        self.meta_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Creating the inductive loss network encoder,
        # M_{\phi_{inductive}}({fx, y, L}).
        self.inductive_dim = 2 * (num_ways * (num_ways * num_shots)) + num_ways * num_shots
        self.inductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.inductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the transductive loss network encoder,
        # M_{\phi_{transductive}}({fx, y^, L}).
        self.transductive_dim = 2 * (num_ways * (num_ways * test_shots)) + num_ways * test_shots
        self.transductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.transductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the task loss network encoder,
        # M_{\phi_{task}}({mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2}).
        self.task_dim = 4 * _calculate_num_layers(base_model)
        self.task_network = torch.nn.Sequential(
            torch.nn.Linear(self.task_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the mixer network for combining the output of the encoders.
        self.mixer_network = torch.nn.Sequential(
            torch.nn.Linear(150, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

    def forward(self, X_support, y_support, X_query, y_query, base_model):

        task_loss = self.meta_loss_fn(base_model(X_support), y_support).mean()

        # Computing the inductive embedding.
        inductive_embedding = _calculate_inductive_embedding(
            X_support, y_support, base_model, self.meta_loss_fn, self.inductive_network,
            self.num_ways, self.num_shots, reduction=False
        )

        # Computing the transductive embedding.
        transductive_embedding = _calculate_transductive_embedding(
            X_support, y_support, X_query, y_query, base_model, self.meta_loss_fn,
            self.transductive_network, self.num_ways, self.num_shots, reduction=False
        )

        # Computing the task embedding.
        task_embedding = _calculate_task_embedding(
            base_model, self.task_network, reduction=False
        )

        # Combining the embeddings and producing the learned loss.
        learned_loss = self.mixer_network(torch.cat((inductive_embedding, transductive_embedding, task_embedding)))
        return learned_loss + task_loss


class LearnedLossV3(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, meta_loss_fn, **kwargs):
        super(LearnedLossV3, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.meta_loss_fn = copy.deepcopy(meta_loss_fn)
        self.meta_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Creating the inductive loss network encoder,
        # Σ M_{\phi_{inductive}}({fx, y, L}).
        self.inductive_dim = 2 * num_ways + 1
        self.inductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.inductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the transductive loss network encoder,
        # Σ M_{\phi_{transductive}}({fx, y^, L}).
        self.transductive_dim = 2 * num_ways + 1
        self.transductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.transductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the task loss network encoder,
        # M_{\phi_{task}}({mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2}).
        self.task_dim = 4 * _calculate_num_layers(base_model)
        self.task_network = torch.nn.Sequential(
            torch.nn.Linear(self.task_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the mixer network for combining the output of the encoders.
        self.mixer_network = torch.nn.Sequential(
            torch.nn.Linear(150, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

    def forward(self, X_support, y_support, X_query, y_query, base_model):

        task_loss = self.meta_loss_fn(base_model(X_support), y_support).mean()

        # Computing the inductive embedding.
        inductive_embedding = _calculate_inductive_embedding(
            X_support, y_support, base_model, self.meta_loss_fn, self.inductive_network,
            self.num_ways, self.num_shots, reduction=True
        )

        # Computing the transductive embedding.
        transductive_embedding = _calculate_transductive_embedding(
            X_support, y_support, X_query, y_query, base_model, self.meta_loss_fn,
            self.transductive_network, self.num_ways, self.num_shots, reduction=True
        )

        # Computing the task embedding.
        task_embedding = _calculate_task_embedding(
            base_model, self.task_network, reduction=False
        )

        # Combining the embeddings and producing the learned loss.
        learned_loss = self.mixer_network(torch.cat((inductive_embedding, transductive_embedding, task_embedding)))
        return learned_loss + task_loss


class LearnedLossV4(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, meta_loss_fn, **kwargs):
        super(LearnedLossV4, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.meta_loss_fn = copy.deepcopy(meta_loss_fn)
        self.meta_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Creating the inductive loss network encoder,
        # Σ M_{\phi_{inductive}}({fx, y, L}).
        self.inductive_dim = 2 * num_ways + 1
        self.inductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.inductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

        # Creating the transductive loss network encoder,
        # Σ M_{\phi_{transductive}}({fx, y^, L}).
        self.transductive_dim = 2 * num_ways + 1
        self.transductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.transductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

        # Creating the task loss network encoder,
        # M_{\phi_{task}}({mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2}).
        self.task_dim = 4 * _calculate_num_layers(base_model)
        self.task_network = torch.nn.Sequential(
            torch.nn.Linear(self.task_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

        # Creating the mixer network for combining the output of the encoders.
        self.mixer_network = torch.nn.Sequential(
            torch.nn.Linear(150, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

    def forward(self, X_support, y_support, X_query, y_query, base_model):

        task_loss = self.meta_loss_fn(base_model(X_support), y_support).mean()

        # Computing the inductive embedding.
        inductive_embedding = _calculate_inductive_embedding(
            X_support, y_support, base_model, self.meta_loss_fn, self.inductive_network,
            self.num_ways, self.num_shots, reduction=True
        )

        # Computing the transductive embedding.
        transductive_embedding = _calculate_transductive_embedding(
            X_support, y_support, X_query, y_query, base_model, self.meta_loss_fn,
            self.transductive_network, self.num_ways, self.num_shots, reduction=True
        )

        # Computing the task embedding.
        task_embedding = _calculate_task_embedding(
            base_model, self.task_network, reduction=False
        )

        # Combining the embeddings and producing the learned loss.
        return inductive_embedding + transductive_embedding + task_embedding + task_loss


class LearnedLossV5(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, meta_loss_fn, **kwargs):
        super(LearnedLossV5, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.meta_loss_fn = copy.deepcopy(meta_loss_fn)
        self.meta_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Creating the inductive loss network encoder,
        # Σ M_{\phi_{inductive}}({fx, y, L}).
        self.inductive_dim = 2 * num_ways + 1
        self.inductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.inductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the transductive loss network encoder,
        # Σ M_{\phi_{transductive}}({fx, y^, L}).
        self.transductive_dim = 2 * num_ways + 1
        self.transductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.transductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the task loss network encoder,
        # Σ M_{\phi_{task}}({mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2}).
        self.task_dim = 4
        self.task_network = torch.nn.Sequential(
            torch.nn.Linear(self.task_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50)
        )

        # Creating the mixer network for combining the output of the encoders.
        self.mixer_network = torch.nn.Sequential(
            torch.nn.Linear(150, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

    def forward(self, X_support, y_support, X_query, y_query, base_model):

        task_loss = self.meta_loss_fn(base_model(X_support), y_support).mean()

        # Computing the inductive embedding.
        inductive_embedding = _calculate_inductive_embedding(
            X_support, y_support, base_model, self.meta_loss_fn, self.inductive_network,
            self.num_ways, self.num_shots, reduction=True
        )

        # Computing the transductive embedding.
        transductive_embedding = _calculate_transductive_embedding(
            X_support, y_support, X_query, y_query, base_model, self.meta_loss_fn,
            self.transductive_network, self.num_ways, self.num_shots, reduction=True
        )

        # Computing the task embedding.
        task_embedding = _calculate_task_embedding(
            base_model, self.task_network, reduction=True
        )

        # Combining the embeddings and producing the learned loss.
        learned_loss = self.mixer_network(torch.cat((inductive_embedding, transductive_embedding, task_embedding)))
        return learned_loss + task_loss


class LearnedLossV6(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, meta_loss_fn, **kwargs):
        super(LearnedLossV6, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.meta_loss_fn = copy.deepcopy(meta_loss_fn)
        self.meta_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Creating the inductive loss network encoder,
        # Σ M_{\phi_{inductive}}({fx, y, L}).
        self.inductive_dim = 2 * num_ways + 1
        self.inductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.inductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

        # Creating the transductive loss network encoder,
        # Σ M_{\phi_{transductive}}({fx, y^, L}).
        self.transductive_dim = 2 * num_ways + 1
        self.transductive_network = torch.nn.Sequential(
            torch.nn.Linear(self.transductive_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

        # Creating the task loss network encoder,
        # Σ M_{\phi_{task}}({mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2}).
        self.task_dim = 4
        self.task_network = torch.nn.Sequential(
            torch.nn.Linear(self.task_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

        # Creating the mixer network for combining the output of the encoders.
        self.mixer_network = torch.nn.Sequential(
            torch.nn.Linear(150, 50),
            _SmoothLeakyReLU(beta=5, gamma=0.25),
            torch.nn.Linear(50, 1)
        )

    def forward(self, X_support, y_support, X_query, y_query, base_model):

        task_loss = self.meta_loss_fn(base_model(X_support), y_support).mean()

        # Computing the inductive embedding.
        inductive_embedding = _calculate_inductive_embedding(
            X_support, y_support, base_model, self.meta_loss_fn, self.inductive_network,
            self.num_ways, self.num_shots, reduction=True
        )

        # Computing the transductive embedding.
        transductive_embedding = _calculate_transductive_embedding(
            X_support, y_support, X_query, y_query, base_model, self.meta_loss_fn,
            self.transductive_network, self.num_ways, self.num_shots, reduction=True
        )

        # Computing the task embedding.
        task_embedding = _calculate_task_embedding(
            base_model, self.task_network, reduction=True
        )

        # Combining the embeddings and producing the learned loss.
        return inductive_embedding + transductive_embedding + task_embedding + task_loss
