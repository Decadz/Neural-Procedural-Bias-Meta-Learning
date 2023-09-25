import torch
import copy


class SmoothLeakyReLU(torch.nn.Module):

    def __init__(self, beta, gamma, **kwargs):
        super(SmoothLeakyReLU, self).__init__()
        self.gamma = gamma  # Leak hyper-parameter.
        self.beta = beta  # Smoothness hyper-parameter.

    def forward(self, x):
        # Dont call 'torch.log()' directly, else you will get numerical instability.
        return self.gamma * x + (1 - self.gamma) * torch.nn.functional.softplus(x, beta=self.beta)


class LearnedLossNetwork(torch.nn.Module):

    def __init__(self, task_loss_fn, input_dim, reduction="mean", logits_to_prob=True, one_hot_encode=True, **kwargs):
        super(LearnedLossNetwork, self).__init__()

        # Meta-loss functions hyper-parameters.
        self.task_loss_fn = task_loss_fn
        self.input_dim = input_dim
        self.reduction = reduction

        # Transformations to apply to the inputs.
        self.logits_to_prob = logits_to_prob
        self.one_hot_encode = one_hot_encode

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 50, bias=False),
            SmoothLeakyReLU(beta=2, gamma=0.25),
            torch.nn.Linear(50, 50, bias=False),
            SmoothLeakyReLU(beta=2, gamma=0.25),
            torch.nn.Linear(50, 1, bias=False),
        )

    def forward(self, y_pred, y_target):

        # Computing the task loss bias term.
        task_loss = self.task_loss_fn(y_pred, y_target)

        # Transforming the prediction and target vectors.
        y_pred, y_target = self._transform_input(y_pred, y_target)

        if self.input_dim == 1:  # If its a single-output problem.
            loss = self.network(torch.cat((y_pred, y_target), dim=1))
            return task_loss + self._reduce_output(loss)

        else:  # If its a multi-output problem.
            res = []  # Iterating over each output label.
            for i in range(self.input_dim):
                yp = torch.unsqueeze(y_pred[:, i], 1)
                y = torch.unsqueeze(y_target[:, i], 1)
                res.append(self.network(torch.cat((yp, y), dim=1)))

            # Taking the mean across the classes.
            loss = torch.stack(res, dim=0).mean(axis=0)
            return task_loss + self._reduce_output(loss)

    def _transform_input(self, y_pred, y_target):
        if self.logits_to_prob:  # Converting the raw logits into probabilities.
            y_pred = torch.nn.functional.sigmoid(y_pred) if self.input_dim == 1 \
                else torch.nn.functional.softmax(y_pred, dim=1)

        if self.one_hot_encode:  # If the target is not already one-hot encoded.
            y_target = torch.nn.functional.one_hot(y_target, num_classes=self.input_dim)

        return y_pred, y_target

    def _reduce_output(self, loss):
        # Applying the desired reduction operation to the loss vector.
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LossNetworkV1(torch.nn.Module):

    def __init__(self, num_ways, num_shots, test_shots, base_model, task_loss_fn, **kwargs):
        super(LossNetworkV1, self).__init__()

        # Set of hyper-parameters that determine input size.
        self.task_loss_fn = copy.deepcopy(task_loss_fn)
        self.task_loss_fn.reduction = "none"
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Inductive: {y, fx, L} = 2 * (num_ways * num_shots) + num_ways * (num_ways * num_shots)
        # Transductive: {y^, fx, L} = 3 * num_ways * (num_ways * test_shots)
        # Task: {mu(theta), sigma(theta), norm(theta)_1, norm(theta)_2} = 4 * num_layers
        self.input_dim = 2 * (num_ways * num_shots) + num_ways * (num_ways * num_shots) + \
                         3 * num_ways * (num_ways * test_shots) + \
                         4 * self._calculate_num_layers(base_model)

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1, bias=False),
        )

    def forward(self, x_sup, yp_sup, y_sup, x_query, yp_query, base_model):
        x = torch.cat((
            self._calculate_inductive_inputs(yp_sup, y_sup),
            self._calculate_transductive_inputs(x_sup, y_sup, x_query, yp_query),
            self._calculate_task_inputs(base_model)
        ))
        # TODO - Implement standardization.
        # learned_loss = self.network(self._standardize_inputs(x))
        learned_loss = self.network(x)
        task_loss = self.task_loss_fn(yp_sup, y_sup).mean()
        return learned_loss + task_loss

    def _calculate_inductive_inputs(self, yp_sup, y_sup):
        yp_support = yp_sup.view(-1)
        y_support = y_sup.view(-1)
        loss = self.task_loss_fn(yp_sup, y_sup).detach().view(-1)
        return torch.cat((yp_support, y_support, loss))

    def _calculate_transductive_inputs(self, x_sup, y_sup, x_query, yp_query):

        # Calculate distances between test and training samples
        distances = torch.cdist(x_query.view(x_query.size(0), -1), x_sup.view(x_sup.size(0), -1))

        # Computing the distance to the prototypes.
        average_distances = []
        for cls in distances.chunk(self.num_ways, dim=1):
            average_distances.append(cls.mean(dim=1))

        # Stacking it into a matrix num_test_instances * num_ways
        average_distances = torch.stack(average_distances, dim=1)

        # Turning the distances into a probabilistic approximation of the ground truth.
        y_query = torch.nn.functional.softmax(average_distances, dim=1)

        # Computing the l2 loss between the predictions and approximated ground truth.
        loss = torch.nn.functional.mse_loss(yp_query.view(-1), y_query.view(-1), reduction="none")

        # Returning a 1D tensor with {y^, fx, L(y, fx)}.
        return torch.cat((y_query.view(-1), yp_query.view(-1), loss))

    def _calculate_task_inputs(self, base_model):
        x_task = []

        # Iterating over all the modules of the model.
        for module in base_model.modules():

            # If its a trainable linear/conv/batch norm layer record the mean and std.
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                param = torch.cat([p.view(-1) for p in module.parameters()])
                x_task.extend([torch.mean(param), torch.std(param), torch.norm(param, p=1), torch.norm(param, p=2)])

        return torch.tensor(x_task).to("cuda:0")

    def _calculate_num_layers(self, base_model):
        count = 0
        for module in base_model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                count += 1
        return count

    def _standardize_inputs(self):
        return None
