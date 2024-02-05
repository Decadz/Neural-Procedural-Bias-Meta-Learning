import higher
import torch
import tqdm
import copy


def meta_training_npbml(base_model, meta_optimizer, base_optimizer, meta_scheduler, training, validation,
                        meta_gradient_steps, base_gradient_steps, meta_batch_size, meta_loss_function,
                        base_loss_function, performance_metric, verbose, **kwargs):

    # Objects for keeping track of the learning history.
    checkpointer = _StateCheckpointerNPBML(
        base_optimizer, validation, base_gradient_steps, meta_gradient_steps, performance_metric, verbose
    )

    # Performing the meta-training phase using unrolled differentiation to update meta parameters.
    for step in (training_progress := tqdm.tqdm(
            range(meta_gradient_steps), position=0, dynamic_ncols=True,
            disable=True if verbose == 0 else False, leave=False)):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each task in our meta batch compute its base trajectory.
        for i in range(meta_batch_size):

            # Sampling a batch of support and query instances.
            X_support, y_support, X_query, y_query = next(training)

            # Merging the support and query into one batch.
            X_support_query = torch.cat((X_support, X_query), dim=0)

            # Creating a differentiable optimizer and stateless models via PyTorch higher.
            with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                # Resetting the classification head to ensure permutation invariance.
                fmodel.reset_classifier()

                # Taking a predetermined number of inner steps before meta update.
                for inner_step in range(base_gradient_steps):
                    #print("inner_step", inner_step)
                    #print("Current GPU memory allocated:", torch.cuda.memory_allocated() / 1024 ** 2, "MB\n")

                    # Computing the predictions on both the support and query set.
                    fx, z = fmodel(X_support_query)
                    loss_support = base_loss_function(fx, z, y_support, fmodel)  # Finding the loss wrt. support set.
                    diffopt.step(loss_support)  # Update base network weights (theta).

                # Computing the task loss and updating the meta weights.
                yp_query, _ = fmodel(X_query)  # Computing the base network predictions on query.
                loss_query = meta_loss_function(yp_query, y_query)  # Finding the loss wrt. query set.
                loss_query.div_(meta_batch_size)  # Dividing the loss by the batch size.
                loss_query.backward()  # Unrolls through the gradient steps.

        # Update the meta parameters.
        meta_optimizer.step()

        # Updating the meta-scheduler step count.
        if meta_scheduler is not None:
            meta_scheduler.step()

        # Checkpointing the model and returning the validation performance.
        performance = checkpointer.checkpoint(base_model, base_loss_function, step)
        training_progress.set_description("Best: " + str(round(checkpointer.best_performance, 4)) +
                                          " | Current: " + str(round(performance, 4)) + " | Progress")

    # Returning the training history and the best base model and loss function.
    return checkpointer.performance_history, checkpointer.best_model, checkpointer.best_loss_function


def meta_testing_npbml(base_model, base_optimizer, dataset, base_gradient_steps, loss_function,
                       performance_metric, test_tasks, verbose, **kwargs):

    # List for keeping track of the learning history.
    performance_history = []

    for _ in (tqdm.tqdm(range(test_tasks), position=1, dynamic_ncols=True, desc="Validating Performance",
                        disable=True if verbose <= 1 else False, leave=False)):

        # Sampling a batch of support and query instances.
        X_support, y_support, X_query, y_query = next(dataset)

        # Merging the support and query into one batch.
        X_support_query = torch.cat((X_support, X_query), dim=0)

        # Creating a differentiable optimizer and stateless models via PyTorch higher.
        with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False,
                                  track_higher_grads=False) as (fmodel, diffopt):

            # Resetting the classification head to ensure permutation invariance.
            fmodel.reset_classifier()

            # Taking a predetermined number of inner steps before meta update.
            for inner_step in range(base_gradient_steps):

                # Computing the predictions on both the support and query set.
                fx, z = fmodel(X_support_query)
                loss_support = loss_function(fx, z, y_support, fmodel)  # Finding the loss wrt. support set.
                diffopt.step(loss_support)  # Update base network weights (theta).

            # Computing the base network predictions on query.
            with torch.no_grad():
                yp_query, _ = fmodel(X_query)

            # Storing the validation performance history.
            performance_history.append(performance_metric(yp_query, y_query).item())

    # Returning the mean and 95% confidence interval of the performance.
    performance = torch.tensor(performance_history)
    mean = torch.mean(performance).item()
    std = torch.std(performance).item()
    ci = 1.96 * (std / (len(performance) ** 0.5))
    return mean, ci


class _StateCheckpointerNPBML(torch.nn.Module):

    def __init__(self, base_optimizer, dataset, base_gradient_steps, meta_gradient_steps,
                 performance_metric, verbose, test_tasks=600, frequency=500, **kwargs):
        super(_StateCheckpointerNPBML, self).__init__()

        # Settings used for the checkpointing.
        self.base_gradient_steps = base_gradient_steps
        self.meta_gradient_steps = meta_gradient_steps
        self.performance_metric = performance_metric
        self.base_optimizer = base_optimizer
        self.test_tasks = test_tasks
        self.frequency = frequency
        self.verbose = verbose
        self.dataset = dataset

        # Tracking the best base model so far.
        self.best_performance = None
        self.best_loss_function = None
        self.best_model = None

        # List for keeping track of the learning history.
        self.performance_history = []

    def checkpoint(self, base_model, loss_function, step):

        # If step is not in the desired frequency the skip checkpointing.
        if step % self.frequency == 0 or step == self.meta_gradient_steps - 1:

            # Performing the meta-validation stage.
            performance, _ = meta_testing_npbml(
                base_model, self.base_optimizer, self.dataset, self.base_gradient_steps, loss_function,
                self.performance_metric, self.test_tasks, self.verbose
            )

            # If this is the best model so far then cache the model.
            if self.best_model is None or performance < self.best_performance:
                self.best_performance = performance
                self.best_model = copy.deepcopy(base_model)
                self.best_loss_function = copy.deepcopy(loss_function)

            # Mapping the optimizer parameters to the best model parameters.
            if step == self.meta_gradient_steps - 1:
                self.base_optimizer.param_groups[0].update({
                    "params": list(self.best_model.base_parameters())})

            # Keeping track of the learning history.
            self.performance_history.append(performance)

        # Returning the most recent performance.
        return self.performance_history[-1]
