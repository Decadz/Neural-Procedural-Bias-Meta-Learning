import torch


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
