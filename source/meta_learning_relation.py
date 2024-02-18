import torch
import tqdm
import copy


def meta_training_relation(model, optimizer, scheduler, training, validation, num_ways, num_shots,
                           gradient_steps, performance_metric, verbose, **kwargs):

    # Objects for keeping track of the learning history.
    checkpointer = _StateCheckpointer(validation, num_ways, num_shots, performance_metric, verbose)

    # Looping until the maximum number of gradient steps is reached.
    for step in (training_progress := tqdm.tqdm(
            range(gradient_steps), position=0, dynamic_ncols=True,
            disable=True if verbose == 0 else False, leave=False)):

        # Clearing the gradient cache.
        optimizer.zero_grad()

        # Sampling a batch of support and query instances.
        X_support, y_support, X_query, y_query = next(training)

        # Merging the support and query into one batch.
        X_support_query = torch.cat((X_support, X_query), dim=0)

        # Performing inference and computing the relation scores.
        y_pred = model(X_support_query)

        # Performing one-hot encoding on the ground truth label.
        y_query = torch.nn.functional.one_hot(y_query, num_classes=num_ways)

        # Computing the loss between the relation scores and ground truth.
        loss = torch.nn.functional.mse_loss(y_pred, y_query.float())

        # Performing the backward pass and gradient step/update.
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Checkpointing the model and returning the validation performance.
        performance = checkpointer.checkpoint(model, step)

        # Updating the progress bar.
        training_progress.set_description("Best: " + str(round(checkpointer.best_performance, 4)) +
                                          " | Current: " + str(round(performance, 4)) + " | Progress")

    # Returning the training history and the best performing base model.
    return checkpointer.best_model, checkpointer.performance_history


def meta_testing_relation(model, dataset, performance_metric, test_tasks, verbose, **kwargs):

    # List for keeping track of the learning history.
    performance_history = []

    for _ in (tqdm.tqdm(range(test_tasks), position=1, dynamic_ncols=True, desc="Validating Performance",
                        disable=True if verbose <= 1 else False, leave=False)):

        # Sampling a batch of support and query instances.
        X_support, y_support, X_query, y_query = next(dataset)

        # Merging the support and query into one batch.
        X_support_query = torch.cat((X_support, X_query), dim=0)

        # Performing inference and computing the relation scores.
        with torch.no_grad():
            y_pred = model(X_support_query)

        # Storing the validation performance history.
        performance_history.append(performance_metric(y_pred, y_query).item())

    # Returning the mean and 95% confidence interval of the performance.
    performance = torch.tensor(performance_history)
    mean = torch.mean(performance).item()
    std = torch.std(performance).item()
    ci = 1.96 * (std / (len(performance) ** 0.5))
    return mean, ci


class _StateCheckpointer(torch.nn.Module):

    def __init__(self, dataset, num_ways, num_shots, performance_metric,
                 verbose, test_tasks=600, frequency=500, **kwargs):

        super(_StateCheckpointer, self).__init__()

        # Dataset configurations for evaluating.
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots

        # Settings used for the checkpointing.
        self.performance_metric = performance_metric
        self.test_tasks = test_tasks
        self.frequency = frequency
        self.verbose = verbose

        # Tracking the best base model so far.
        self.best_performance = None
        self.best_model = None

        # List for keeping track of the learning history.
        self.performance_history = []

    def checkpoint(self, base_model, step):

        # If step is not in the desired frequency the skip checkpointing.
        if step % self.frequency == 0:

            # Performing the meta-validation stage.
            performance, _ = meta_testing_relation(
                base_model, self.dataset, self.performance_metric, self.test_tasks, self.verbose
            )

            # If this is the best model so far then cache the model.
            if self.best_model is None or performance < self.best_performance:
                self.best_performance = performance
                self.best_model = copy.deepcopy(base_model)

            # Keeping track of the learning history.
            self.performance_history.append(performance)

        # Returning the most recent performance.
        return self.performance_history[-1]
