import torch
import tqdm
import copy


def pretraining(model, optimizer, scheduler, training, validation, num_ways, num_shots, gradient_steps,
                batch_size, loss_function, performance_metric, verbose, device, **kwargs):

    """
    A vanilla training loop which uses stochastic gradient descent to learn
    i.e., preform pretraining, of the provided base network.

    :param model: Base network used for the given task.
    :param num_ways: Number of classes (FSL ways).
    :param num_shots: Number of training (support) instances.
    :param optimizer: Backpropagation gradient optimizer.
    :param scheduler: PyTorch learning rate scheduler.
    :param training: PyTorch Dataset containing the training dataset.
    :param validation: FSL Dataset containing the validation dataset.
    :param gradient_steps: Number of maximum gradient steps.
    :param batch_size: Backpropagation batch size.
    :param loss_function: Loss function to minimize.
    :param performance_metric: Performance metric to use for evaluation.
    :param verbose: Display console output at different levels {0, 1}.
    :param device: Device used for Pytorch related computation.
    :return: List containing the meta-training history.
    """

    # Extracting the custom dataset object into a dataloader object.
    if not isinstance(training.dataset, torch.utils.data.DataLoader):
        training = torch.utils.data.DataLoader(training.dataset, batch_size=batch_size, shuffle=True)

    # Objects for keeping track of the learning history.
    checkpointer = _StateCheckpointer(validation, num_ways, num_shots, performance_metric)

    # A list for keeping track of the fine-tuning training accuracy.
    fine_tuning_history = []

    # Looping until the maximum number of gradient steps is reached.
    for step in (training_progress := tqdm.tqdm(
            range(gradient_steps), position=0, dynamic_ncols=True,
            disable=False if verbose >= 1 else True, leave=False)):

        # Clearing the gradient cache.
        optimizer.zero_grad()

        # Sampling a mini batch from the task.
        X, y = next(iter(training))
        X, y = X.to(device), y.to(device)

        # Performing inference and computing the loss.
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # Performing the backward pass and gradient step/update.
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Resetting any batch normalization layer.
        model.reset_batch_norm()

        # Checkpointing the model and returning the validation performance.
        performance = checkpointer.checkpoint(model, step)
        training_progress.set_description("Performance " + str(round(performance, 4)))

        # Recording the fine-tuning accuracy.
        performance = performance_metric(y_pred, y).item()
        fine_tuning_history.append(performance)

    # Returning the training history and the best performing base model.
    return checkpointer.best_model, checkpointer.performance_history, fine_tuning_history


class _StateCheckpointer(torch.nn.Module):

    def __init__(self, dataset, num_ways, num_shots, performance_metric,
                 test_tasks=600, frequency=500, **kwargs):

        super(_StateCheckpointer, self).__init__()

        # Dataset configurations for evaluating.
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots

        # Settings used for the checkpointing.
        self.performance_metric = performance_metric
        self.test_tasks = test_tasks
        self.frequency = frequency

        # Tracking the best base model so far.
        self.best_performance = None
        self.best_model = None

        # List for keeping track of the learning history.
        self.performance_history = []

    def checkpoint(self, base_model, step):

        # If step is not in the desired frequency the skip checkpointing.
        if step % self.frequency == 0:

            # Sampling a batch of support and query instances.
            X_support, y_support, X_query, y_query = next(self.dataset)

            # Computing the support and query embeddings.
            support = base_model.encoder(X_support)
            query = base_model.encoder(X_query)

            # Computing the prototypes for each of the ways (classes).
            prototypes = support.reshape(self.num_shots, self.num_ways, -1).mean(dim=0)

            # Computing each query instances distance to each of the prototypes.
            logits = torch.mm(query, torch.nn.functional.normalize(prototypes, p=2, dim=-1).t())

            # Computing the performance with the given performance metric.
            performance = self.performance_metric(logits, y_query).item()

            # If this is the best model so far then cache the model.
            if self.best_model is None or performance < self.best_performance:
                self.best_performance = performance
                self.best_model = copy.deepcopy(base_model)

            # Keeping track of the learning history.
            self.performance_history.append(performance)

        return self.best_performance
