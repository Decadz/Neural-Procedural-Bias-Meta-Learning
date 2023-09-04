import torch
import tqdm


def backpropagation(model, optimizer, scheduler, task, gradient_steps, batch_size, loss_function,
                    performance_metric, verbose, device, **kwargs):

    """
    A vanilla training loop which uses stochastic gradient descent to learn the
    parameters of the base network.

    :param model: Base network used for the given task.
    :param optimizer: Backpropagation gradient optimizer.
    :param scheduler: PyTorch learning rate scheduler.
    :param task: PyTorch Dataset containing the training data.
    :param gradient_steps: Number of maximum gradient steps.
    :param batch_size: Backpropagation batch size.
    :param loss_function: Loss function to minimize.
    :param performance_metric: Performance metric to use for evaluation.
    :param verbose: Display console output at different levels {0, 1}.
    :param device: Device used for Pytorch related computation.
    :return: List containing the meta-training history.
    """

    if not isinstance(task, torch.utils.data.DataLoader):
        task = torch.utils.data.DataLoader(task, batch_size=batch_size, shuffle=True)

    training_history = []

    # Looping until the maximum number of gradient steps is reached.
    for step in (training_progress := tqdm.tqdm(
            range(gradient_steps), position=0, dynamic_ncols=True,
            disable=False if verbose >= 1 else True, leave=False)):

        # If the predetermined number of gradient steps has been reached.
        if step >= gradient_steps:
            break

        # Clearing the gradient cache.
        optimizer.zero_grad()

        # Sampling a mini batch from the task.
        X, y = next(iter(task))
        X, y = X.to(device), y.to(device)

        # Performing inference and computing the loss.
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # Performing the backward pass and gradient step/update.
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Recording the training performance and updating the progression bar.
        performance = performance_metric(y_pred, y).item()
        training_history.append(performance)
        training_progress.set_description("Performance " + str(round(performance, 4)))

    return training_history


def evaluate(model, task, device, performance_metric, batch_size=100, **kwargs):

    """
    Performs inference on the provided model, and computes the
    performance using the provided performance metric.

    :param model: Base network used for the given task.
    :param task: PyTorch DataLoader used for evaluation.
    :param device: Device used for Pytorch related computation.
    :param performance_metric: Performance metric to use for evaluation.
    :param batch_size: Batch size used for inference.
    :return performance: Final inference performance on task.
    """

    # Creating a PyTorch dataloader object for generating batches.
    task = torch.utils.data.DataLoader(task, batch_size=batch_size, shuffle=False)
    pred_labels, true_labels = [], []

    model.eval()  # Switching network to inference mode.
    with torch.no_grad():  # Disabling gradient calculations.

        # Iterating over the whole dataset in batches.
        for instances, labels in task:
            yp = model(instances.to(device))
            pred_labels.append(yp)
            true_labels.append(labels.to(device))

    # Converting the list to a PyTorch tensor.
    pred_labels = torch.cat(pred_labels)
    true_labels = torch.cat(true_labels)

    model.train()  # Switching network back to training mode.

    # Returning the performance of the trained model.
    return performance_metric(pred_labels, true_labels).item()
