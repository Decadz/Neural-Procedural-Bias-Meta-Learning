import higher
import torch
import tqdm
import copy


def meta_training(base_model, meta_optimizer, base_optimizer, base_bootstrapped_optimizer, meta_scheduler,
                  training, validation, meta_gradient_steps, base_gradient_steps, base_bootstrapped_gradient_steps,
                  meta_batch_size, meta_loss_function, base_loss_function, matching_function, performance_metric,
                  verbose, **kwargs):

    # Objects for keeping track of the learning history.
    checkpointer = _StateCheckpointer(
        base_optimizer, validation, base_gradient_steps, meta_gradient_steps, performance_metric
    )

    # Performing the meta-training phase using unrolled differentiation to update meta parameters.
    for step in (training_progress := tqdm.tqdm(
            range(meta_gradient_steps), position=0, dynamic_ncols=True,
            disable=False if verbose >= 1 else True, leave=False)):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each task in our meta batch compute its base trajectory.
        for i in range(meta_batch_size):

            # Sampling a batch of support and query instances.
            X_support, y_support, X_query, y_query = next(training)

            # Creating a differentiable optimizer and stateless models via PyTorch higher.
            with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                # Resetting any batch normalization layer.
                fmodel.reset_batch_norm()

                # Taking a predetermined number of inner steps before meta update.
                for _ in range(base_gradient_steps):

                    # Computing the loss using the learned loss and updating the base weights.
                    yp_support = fmodel(X_support)  # Computing the base network predictions on support.
                    loss_support = base_loss_function(yp_support, y_support)  # Finding the loss wrt. support set.

                    # TODO - TESTING CODE
                    #yp_query = fmodel(X_query)
                    #loss_support = base_loss_function(X_support, yp_support, y_support, X_query, yp_query.detach(), fmodel)
                    # TODO - TESTING CODE

                    diffopt.step(loss_support)  # Update base network weights (theta).

                # Perform the typical unrolled differentiation objective.
                if base_bootstrapped_gradient_steps == 0:

                    # Computing the task loss and updating the meta weights.
                    yp_query = fmodel(X_query)  # Computing the base network predictions on query.
                    loss_query = meta_loss_function(yp_query, y_query)  # Finding the loss wrt. query set.
                    loss_query.div_(meta_batch_size)  # Dividing the loss by the batch size.
                    loss_query.backward()  # Unrolls through the gradient steps.

                else:
                    # Creating a copy of the base model for generating a bootstrapping target.
                    bootstrapped_model = copy.deepcopy(base_model)
                    bootstrapped_model.load_state_dict(copy.deepcopy(fmodel.state_dict()))
                    bootstrapped_optimizer = base_bootstrapped_optimizer(bootstrapped_model.adapt_parameters())

                    # Taking a predetermined number of bootstrapping steps.
                    for _ in range(base_bootstrapped_gradient_steps):
                        bootstrapped_optimizer.zero_grad()  # Clearing out the gradient cache.
                        yp_query = bootstrapped_model(X_query)  # Computing the base network predictions on query.
                        loss_query = meta_loss_function(yp_query, y_query)  # Finding the loss wrt. query set.
                        loss_query.backward()  # Computing the gradients wrt. to the loss.
                        bootstrapped_optimizer.step()  # Updating the model parameters.

                    # Performing the meta-update by using a matching function.
                    target_output = torch.nn.utils.parameters_to_vector(fmodel.adapt_parameters())
                    target_bootstrapped = torch.nn.utils.parameters_to_vector(bootstrapped_model.adapt_parameters())
                    task_loss = matching_function(target_output, target_bootstrapped)
                    task_loss.div_(meta_batch_size)
                    task_loss.backward()

        # Applying meta-gradient clipping as done in MAML++.
        torch.nn.utils.clip_grad_value_(base_model.parameters(), clip_value=10)

        # Update the meta parameters.
        meta_optimizer.step()

        # Updating the meta-scheduler step count.
        if meta_scheduler is not None:
            meta_scheduler.step()

        # Checkpointing the model and returning the validation performance.
        performance = checkpointer.checkpoint(base_model, base_loss_function, step)
        training_progress.set_description("Performance " + str(round(performance, 4)))

    # Returning the training history and the best base model and loss function.
    return checkpointer.performance_history, checkpointer.best_model, checkpointer.best_loss_function


def meta_testing(base_model, base_optimizer, dataset, base_gradient_steps, loss_function,
                 performance_metric, test_tasks, verbose, **kwargs):

    # List for keeping track of the learning history.
    performance_history = []

    for _ in range(test_tasks):

        # Sampling a batch of support and query instances.
        X_support, y_support, X_query, y_query = next(dataset)

        # Creating a differentiable optimizer and stateless models via PyTorch higher.
        with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False,
                                  track_higher_grads=False) as (fmodel, diffopt):

            # Resetting any batch normalization layer.
            fmodel.reset_batch_norm()

            # Taking a predetermined number of inner steps before meta update.
            for _ in range(base_gradient_steps):

                # Computing the loss using the learned loss and updating the base weights.
                yp_support = fmodel(X_support)  # Computing the base network predictions on support.
                loss_support = loss_function(yp_support, y_support)  # Finding the loss wrt. support set.
                diffopt.step(loss_support)  # Update base network weights (theta).

            # Computing the task loss and updating the meta weights.
            yp_query = fmodel(X_query)  # Computing the base network predictions on query.

            # Storing the validation performance history.
            performance_history.append(performance_metric(yp_query, y_query).item())

    # Returning the mean and standard deviation of performance.
    performance = torch.tensor(performance_history)
    return torch.mean(performance).item(), torch.std(performance).item()


class _StateCheckpointer(torch.nn.Module):

    def __init__(self, base_optimizer, dataset, base_gradient_steps, meta_gradient_steps,
                 performance_metric, test_tasks=600, frequency=500, **kwargs):
        super(_StateCheckpointer, self).__init__()

        # Settings used for the checkpointing.
        self.base_gradient_steps = base_gradient_steps
        self.meta_gradient_steps = meta_gradient_steps
        self.performance_metric = performance_metric
        self.base_optimizer = base_optimizer
        self.test_tasks = test_tasks
        self.frequency = frequency
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
            performance, _ = meta_testing(
                base_model, self.base_optimizer, self.dataset, self.base_gradient_steps,
                loss_function, self.performance_metric, self.test_tasks, 0
            )

            # If this is the best model so far then cache the model.
            if self.best_model is None or performance < self.best_performance:
                self.best_performance = performance
                self.best_loss_function = copy.deepcopy(loss_function)
                self.best_model = copy.deepcopy(base_model)

            # Mapping the optimizer parameters to the best model parameters.
            if step == self.meta_gradient_steps - 1:
                self.base_optimizer.param_groups[0].update({"params": list(self.best_model.parameters())})

            # Keeping track of the learning history.
            self.performance_history.append(performance)

        return self.best_performance
