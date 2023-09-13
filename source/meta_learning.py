import higher
import torch
import tqdm


def meta_training(base_model, meta_optimizer, base_optimizer, meta_scheduler, training, validation,
                  meta_gradient_steps, base_gradient_steps, meta_batch_size, loss_function,
                  performance_metric, verbose, **kwargs):

    # List for keeping track of the learning history.
    training_history = []

    # Performing the meta-training phase using unrolled differentiation to update meta parameters.
    for step in (training_progress := tqdm.tqdm(
            range(meta_gradient_steps), position=0, dynamic_ncols=True,
            disable=False if verbose >= 1 else True, leave=False)):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # List for keeping track of the learning history.
        performance_history = []

        # For each task in our meta batch compute its base trajectory.
        for i in range(meta_batch_size):

            # Sampling a batch of support and query instances.
            X_support, y_support, X_query, y_query = next(training)

            # Creating a differentiable optimizer and stateless models via PyTorch higher.
            with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                # Taking a predetermined number of inner steps before meta update.
                for _ in range(base_gradient_steps):

                    # Computing the loss using the learned loss and updating the base weights.
                    yp_support = fmodel(X_support)  # Computing the base network predictions on support.
                    loss_support = loss_function(yp_support, y_support)  # Finding the loss wrt. support set.
                    diffopt.step(loss_support)  # Update base network weights (theta).

                # Computing the task loss and updating the meta weights.
                yp_query = fmodel(X_query)  # Computing the base network predictions on query.
                loss_query = loss_function(yp_query, y_query)  # Finding the loss wrt. query set.
                loss_query.backward()  # Unrolls through the gradient steps.

                # Storing the validation performance history.
                performance_history.append(performance_metric(yp_query, y_query).item())

        meta_optimizer.step()  # Update the meta parameters.

        # Updating the meta-scheduler step count.
        if meta_scheduler is not None:
            meta_scheduler.step()

        # Updating training history and progression bar.
        training_history.append(sum(performance_history)/len(performance_history))
        performance = sum(performance_history)/len(performance_history)
        training_progress.set_description("Performance " + str(round(performance, 4)))

    return training_history


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
