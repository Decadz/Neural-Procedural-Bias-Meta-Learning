import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits
import higher
import random
import torch
import tqdm
import copy
import os


def main():

    # Path to save the figures and animations.
    path = "../experiments/results/synthetic/"
    if not os.path.exists(path):
        os.makedirs(path)

    # Setting the reproducibility seed in PyTorch.
    torch.cuda.manual_seed_all(0)
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Meta-learning the loss function, infusing extra information in two different ways.
    #warp_optimizer = learned_warp_optimizer(func)
    #ml3_loss = learned_loss_function(func)
    #npbml_optimizer_1, npbml_loss_1 = learned_procedural_biases(func)
    npbml_optimizer_2, npbml_loss_2 = learned_procedural_biases_extra1(func)
    npbml_optimizer_3 = learned_procedural_biases_extra2(func)
    npbml_loss_4 = learned_procedural_biases_extra3(func)
    #npbml_optimizer_3, npbml_loss_3 = learned_procedural_biases_bmg(func)


    # Performing the meta testing phase on the following seeds (selected because they look nice).
    for seed in range(50):
        print("Starting:", str(seed))

        # Setting the reproducibility seed in PyTorch.
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Generating the optimization problem and problem initialization.
        s, a, b = torch.randint(1, 11, (1,)), torch.randint(-1, 2, (3,)), torch.randint(-5, 6, (3,))
        x1, x2 = random.uniform(-3, 3), random.uniform(-3, 3)

        sgd_trajectory = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s)
        """
        warp_trajectory = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s, warp_fn=warp_optimizer)

        ml3_trajectory = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s, loss_fn=ml3_loss)

        npbml_trajectory_1 = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s,
            warp_fn=npbml_optimizer_1, loss_fn=npbml_loss_1)
        """

        npbml_trajectory_2 = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s,
            warp_fn=npbml_optimizer_2, loss_fn=npbml_loss_2)

        npbml_trajectory_3 = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s, warp_fn=npbml_optimizer_3)

        npbml_trajectory_4 = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s, loss_fn=npbml_loss_4)

        # Setting up the parameters for plotting different (multi) surfaces.
        #paths = [sgd_trajectory["true"],  warp_trajectory["warp"], ml3_trajectory["loss"], npbml_trajectory_1["loss"], npbml_trajectory_2["loss"]]
        #loss_fns = [None, None, ml3_loss, npbml_loss_1, npbml_loss_2]
        #optimizers = [None, warp_optimizer, None, npbml_optimizer_1, npbml_optimizer_2]

        paths = [sgd_trajectory["true"], npbml_trajectory_2["loss"], npbml_trajectory_3["warp"], npbml_trajectory_4["loss"]]
        loss_fns = [None, npbml_loss_2, None, npbml_loss_4]
        optimizers = [None, npbml_optimizer_2, npbml_optimizer_3, None]

        # Plotting each of the loss landscapes, and then the trajectories on those landscapes.
        plot_landscape_3d_multi(func, a, b, s, paths, loss_fns, optimizers, True, path + "3d-multi-" + str(seed))
        plot_landscape_2d_multi(func, a, b, s, paths, loss_fns, optimizers, True, path + "2d-multi-" + str(seed))
        animate_landscape_3d_multi(func, a, b, s, paths, loss_fns, optimizers, True, path + "3d-multi-" + str(seed))
        animate_landscape_2d_multi(func, a, b, s, paths, loss_fns, optimizers, True, path + "2d-multi-" + str(seed))

        # Setting up the parameters for plotting on the same true (single) surfaces.
        #paths = [sgd_trajectory["true"], warp_trajectory["true"], ml3_trajectory["true"], npbml_trajectory_1["true"], npbml_trajectory_2["true"]]
        paths = [sgd_trajectory["true"], npbml_trajectory_2["true"], npbml_trajectory_3["true"]]

        # Plotting the true landscapes, and then the trajectories on that single landscapes.
        plot_landscape_3d_single(func, a, b, s, paths, True, path + "3d-single-" + str(seed))
        plot_landscape_2d_single(func, a, b, s, paths, True, path + "2d-single-" + str(seed))
        animate_landscape_3d_single(func, a, b, s, paths, True, path + "3d-single-" + str(seed))
        animate_landscape_2d_single(func, a, b, s, paths, True, path + "2d-single-" + str(seed))

        print("Finished:", str(seed))


# ============================================================
# Training related code for the different methods.
# ============================================================


def func(x, a, b, s):
    return b[0] * (a[0] - x[0]) ** 2 * torch.exp(-x[0] ** 2 - (x[1] + a[1]) ** 2) \
           - b[1] * (x[0] / s - x[0] ** 3 - x[1] ** 5) * torch.exp(-x[0] ** 2 - x[1] ** 2) \
           - b[2] * torch.exp(-(x[0] + a[2]) ** 2 - x[0] ** 2)


def gradient_descent(x, func, a, b, s, warp_fn=None, loss_fn=None):

    optimizer = torch.optim.SGD([x], lr=0.1)  # Set this to 0.01 to get smoother trajectory.
    true_trajectory, warp_trajectory, loss_trajectory = [], [], []

    for step in range(100):

        y = func(x, a=a, b=b, s=s)
        true_trajectory.append([x[0].item(), x[1].item(), y.item()])

        if warp_fn is not None:
            y = warp_fn(y)
            warp_trajectory.append([x[0].item(), x[1].item(), y.item()])

        if loss_fn is not None:
            y = loss_fn(x, y)
            loss_trajectory.append([x[0].item(), x[1].item(), y.item()])

        # Compute gradients
        y.backward()

        # Update tensor values based on gradients
        optimizer.step()

        # Clear gradients for the next step
        optimizer.zero_grad()

    return {"true": true_trajectory, "warp": warp_trajectory, "loss": loss_trajectory}


class LossNetwork(torch.nn.Module):

    def __init__(self):
        super(LossNetwork, self).__init__()

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3, 50),
            torch.nn.ELU(),
            torch.nn.Linear(50, 50),
            torch.nn.ELU(),
            torch.nn.Linear(50, 1)
        )

    def forward(self, x, y):
        return self.network(torch.cat((x, y), dim=0))


class OptimizerNetwork(torch.nn.Module):

    def __init__(self):
        super(OptimizerNetwork, self).__init__()

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.ELU(),
            torch.nn.Linear(50, 50),
            torch.nn.ELU(),
            torch.nn.Linear(50, 1)
        )

    def forward(self, y):
        return self.network(y)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.s = torch.randint(1, 11, (1,))
        self.a = torch.randint(-1, 2, (3,))
        self.b = torch.randint(-5, 6, (3,))
        self.x = torch.nn.Parameter(torch.rand(2, requires_grad=True) * 6 - 3)

    def forward(self, func):
        return func(self.x, self.a, self.b, self.s)


def learned_initialization(func):

    """
    Code for meta-learning a parameter intialization, which uses "MAML", a technique
    presented by Finn, C. et al. "Model-Agnostic Meta-Learning for Fast Adaptation
    of Deep Networks". ICML2017.

    :param func: Function that we are trying to minimize.
    """

    meta_model = Model()

    # Defining the outer optimizer for the meta-loss network.
    meta_optimizer = torch.optim.Adam([meta_model.x], lr=0.001)

    # Performing meta-training over a number of meta-gradient steps.
    for step in tqdm.tqdm(range(200), desc="MAML"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_model.x = torch.nn.Parameter(meta_model.x.clone().detach())
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    diffopt.step(y)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                y = fmodel(func)  # Finding the loss wrt. meta (task) loss.
                y.backward()  # Accumulates gradients wrt. to meta parameters.

        # Updating meta weights (phi).
        meta_optimizer.step()

    return meta_model.x


def learned_warp_optimizer(func):

    """
    Code for meta-learning a preconditioning optimizer, which uses "WarpGrad", a technique
    presented by Flennerhag, S. "Meta-Learning with Warped Gradient Descent". ICLR2020. 
    Note we use simple unrolled differentiation instead of the Warp-Leap objective for 
    simplicity.

    :param func: Function that we are trying to minimize.
    """

    meta_network = OptimizerNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_optimizer = torch.optim.Adam(meta_network.parameters(), lr=0.001)

    # Performing meta-training over a number of meta-gradient steps.
    for step in tqdm.tqdm(range(200), desc="WarpGrad"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    warp_y = meta_network(y)  # Warping the loss via a learned optimizer.
                    diffopt.step(warp_y)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                y = fmodel(func)  # Finding the loss wrt. meta (task) loss.
                y.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        # Updating meta weights (phi).
        meta_optimizer.step()

    return meta_network


def learned_loss_function(func):

    """
    Code for meta-learning a loss function, which uses "ML3", a technique presented 
    by Bechtle, et al. “Meta-Learning via Learned Loss”. ICPR2021

    :param func: Function that we are trying to minimize.
    """

    meta_network = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_optimizer = torch.optim.Adam(meta_network.parameters(), lr=0.001)

    # Performing meta-training over a number of meta-gradient steps.
    for step in tqdm.tqdm(range(200), desc="ML3"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    # Computing the loss using the learned loss and updating the base weights.
                    y = fmodel(func)
                    base_loss = meta_network(fmodel.x, y)
                    diffopt.step(base_loss)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                y = fmodel(func)  # Finding the loss wrt. meta (task) loss.
                y.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        # Updating meta weights (phi).
        meta_optimizer.step()

    return meta_network


def learned_procedural_biases(func):

    learned_optimizer = OptimizerNetwork()
    learned_loss = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_param = [
        {"params": learned_optimizer.parameters(), "lr": 0.001},
        {"params": learned_loss.parameters(), "lr": 0.001}
    ]

    meta_optimizer = torch.optim.Adam(meta_param, lr=0.001)

    # Performing meta-training over a number of meta-gradient steps.
    for step in tqdm.tqdm(range(200), desc="NPBML"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    warp_y = learned_optimizer(y)  # Warping the loss via a learned optimizer.
                    base_loss = learned_loss(fmodel.x, warp_y)
                    diffopt.step(base_loss)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                y = fmodel(func)  # Finding the loss wrt. meta (task) loss.
                y.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        # Updating meta weights (phi).
        meta_optimizer.step()

    return learned_optimizer, learned_loss


def learned_procedural_biases_extra1(func):

    learned_optimizer = OptimizerNetwork()
    learned_loss = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_param = [
        {"params": learned_optimizer.parameters(), "lr": 0.001},
        {"params": learned_loss.parameters(), "lr": 0.001}
    ]

    meta_optimizer = torch.optim.Adam(meta_param, lr=0.001)

    # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
    for step in tqdm.tqdm(range(500), desc="NPBML+Extra"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Approximating the ground truth minimum location x using random search
            rs = (torch.rand((2, 1000), requires_grad=True) * 6 - 3).detach().requires_grad_(False)
            rs_results = func(rs, base_model.a, base_model.b, base_model.s)
            x_opt = torch.tensor([rs[0][torch.argmin(rs_results)], rs[1][torch.argmin(rs_results)]])

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    warp_y = learned_optimizer(y)  # Warping the loss via a learned optimizer.
                    base_loss = learned_loss(fmodel.x, warp_y)
                    diffopt.step(base_loss)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                task_loss = ((x_opt - fmodel.x) ** 2).sum()  # Loss to the (approx) optimal solution.
                task_loss.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        # Updating meta weights (phi).
        meta_optimizer.step()

    return learned_optimizer, learned_loss


def learned_procedural_biases_extra2(func):

    learned_optimizer = OptimizerNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_param = [
        {"params": learned_optimizer.parameters(), "lr": 0.001}
    ]

    meta_optimizer = torch.optim.Adam(meta_param, lr=0.001)

    # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
    for step in tqdm.tqdm(range(500), desc="NPBML+Extra"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Approximating the ground truth minimum location x using random search
            rs = (torch.rand((2, 1000), requires_grad=True) * 6 - 3).detach().requires_grad_(False)
            rs_results = func(rs, base_model.a, base_model.b, base_model.s)
            x_opt = torch.tensor([rs[0][torch.argmin(rs_results)], rs[1][torch.argmin(rs_results)]])

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    warp_y = learned_optimizer(y)  # Warping the loss via a learned optimizer.
                    diffopt.step(warp_y)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                task_loss = ((x_opt - fmodel.x) ** 2).sum()  # Loss to the (approx) optimal solution.
                task_loss.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        # Updating meta weights (phi).
        meta_optimizer.step()

    return learned_optimizer


def learned_procedural_biases_extra3(func):

    learned_loss = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_param = [
        {"params": learned_loss.parameters(), "lr": 0.001}
    ]

    meta_optimizer = torch.optim.Adam(meta_param, lr=0.001)

    # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
    for step in tqdm.tqdm(range(500), desc="NPBML+Extra"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Approximating the ground truth minimum location x using random search
            rs = (torch.rand((2, 1000), requires_grad=True) * 6 - 3).detach().requires_grad_(False)
            rs_results = func(rs, base_model.a, base_model.b, base_model.s)
            x_opt = torch.tensor([rs[0][torch.argmin(rs_results)], rs[1][torch.argmin(rs_results)]])

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    base_loss = learned_loss(fmodel.x, y)
                    diffopt.step(base_loss)  # Update base network weights (theta).

                # Computing the task loss and storing the change to the meta weights.
                task_loss = ((x_opt - fmodel.x) ** 2).sum()  # Loss to the (approx) optimal solution.
                task_loss.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        # Updating meta weights (phi).
        meta_optimizer.step()

    return learned_loss


def learned_procedural_biases_bmg(func):

    learned_optimizer = OptimizerNetwork()
    learned_loss = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_param = [
        {"params": learned_optimizer.parameters(), "lr": 0.001},
        {"params": learned_loss.parameters(), "lr": 0.001}
    ]

    meta_optimizer = torch.optim.Adam(meta_param, lr=0.001)

    # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
    for step in tqdm.tqdm(range(200), desc="NPBML+BMG"):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    y = fmodel(func)  # Calculating the loss at the given point.
                    warp_y = learned_optimizer(y)  # Warping the loss via a learned optimizer.
                    base_loss = learned_loss(fmodel.x, warp_y)
                    diffopt.step(base_loss)  # Update base network weights (theta).

            # Creating a copy of the base model for generating a bootstrapping target.
            bootstrapped_model = copy.deepcopy(base_model)
            bootstrapped_model.load_state_dict(copy.deepcopy(fmodel.state_dict()))
            bootstrapped_optimizer = torch.optim.Adam(bootstrapped_model.parameters(), 0.1)

            # Taking a predetermined number of bootstrapping steps.
            for _ in range(10):
                bootstrapped_optimizer.zero_grad()  # Clearing out the gradient cache.
                y = bootstrapped_model(func)  # Computing the base network predictions on query.
                y.backward()  # Computing the gradients wrt. to the loss.
                bootstrapped_optimizer.step()  # Updating the model parameters.

            # Computing the task loss and storing the change to the meta weights.
            task_loss = ((bootstrapped_model.x - fmodel.x) ** 2).sum()  # Loss to the (approx) optimal solution.
            task_loss.backward()  # Accumulates gradients wrt. to meta parameters.

        # Updating meta weights (phi).
        meta_optimizer.step()

    return learned_optimizer, learned_loss


# ============================================================
# Functions for generating 2D and 3D static visualizations.
# ============================================================


def plot_landscape_3d_multi(func, a, b, s, trajectories, loss_functions, optimizers, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(
        figsize=(len(trajectories) * 7, 7),
        nrows=1, ncols=len(trajectories),
        subplot_kw={"projection": "3d"},
        gridspec_kw={'top': 1, 'bottom': 0, 'wspace': 0, 'hspace': 0}
    )

    # Colours to use for the surfaces and trajectories.
    surface_colors = ["#ad67ca", "#7881c4", "#78b3c4", "#82b1aa", "#6fbc8b"]
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from X and Y values

    for i, (trajectory, loss_function, optimizer) in enumerate(zip(trajectories, loss_functions, optimizers)):

        surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s).flatten()

        if optimizer is not None:
            surface_Z = optimizer(surface_Z.unsqueeze(1)).clone().detach().reshape(surface_X.size()).flatten()

        if loss_function is not None:
            surface_Z = loss_function.network(torch.cat((torch.cat((
                surface_X.reshape(-1, 1), surface_Y.reshape(-1, 1)), dim=1),
                surface_Z.unsqueeze(1)
            ), dim=1))

        # Reshaping and converting into the correct type.
        surface_Z = surface_Z.reshape(surface_X.size()).detach().numpy()

        # Plotting the wireframe, i.e. loss landscape that we are optimizing over.
        axes[i].plot_wireframe(surface_X, surface_Y, surface_Z, linewidth=0.75,
                               edgecolor=surface_colors[i], rstride=10, cstride=10, zorder=0)

        x_trajectory = enforce_input_range([row[0] for row in trajectory])
        y_trajectory = enforce_input_range([row[1] for row in trajectory])
        z_trajectory = [row[2] for row in trajectory]

        axes[i].plot(x_trajectory, y_trajectory, z_trajectory,
                     c=path_colors[i], linewidth=3, zorder=10)

        axes[i].plot(x_trajectory[0], y_trajectory[0], z_trajectory[0],
                     c=path_colors[i], marker="o", markersize=8, zorder=20)

        axes[i].plot(x_trajectory[-1], y_trajectory[-1], z_trajectory[-1],
                     c=path_colors[i], marker="*", markersize=15, zorder=20)

        # Clear the background of figure artifacts.
        clear_background(axes[i])

    if save:  # Saving the file to the given path.
        plt.savefig(file_name + ".pdf")
    else:  # Showing the figure.
        plt.show()

    plt.close()


def plot_landscape_3d_single(func, a, b, s, trajectories, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(figsize=(10, 8), nrows=1, ncols=1, subplot_kw={"projection": "3d"})

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from X and Y values

    # Compute the Z values (function values)
    Z = func([surface_X, surface_Y], a=a, b=b, s=s)

    # Plot the wireframe mesh
    axes.plot_wireframe(surface_X, surface_Y, Z, linewidth=0.75,
                        edgecolor="#7881c4", rstride=10, cstride=10, zorder=0)

    for trajectory, color in zip(trajectories, path_colors):
        x_trajectory = enforce_input_range([row[0] for row in trajectory])
        y_trajectory = enforce_input_range([row[1] for row in trajectory])
        z_trajectory = [row[2] for row in trajectory]

        axes.plot(x_trajectory, y_trajectory, z_trajectory,
                  c=color, linewidth=3, zorder=10)

        axes.plot(x_trajectory[0], y_trajectory[0], z_trajectory[0],
                  c=color, marker="o", markersize=8, zorder=20)

        axes.plot(x_trajectory[-1], y_trajectory[-1], z_trajectory[-1],
                  c=color, marker="*", markersize=15, zorder=20)

    # Clear the background of figure artifacts.
    clear_background(axes)

    if save:  # Saving the file to the given path.
        plt.savefig(file_name + ".pdf")
    else:  # Showing the figure.
        plt.show()

    plt.close()


def plot_landscape_2d_multi(func, a, b, s, trajectories, loss_functions, optimizers, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(
        figsize=(len(trajectories) * 7, 6),
        nrows=1, ncols=len(trajectories)
    )

    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from X and Y values

    for i, (trajectory, loss_function, optimizer) in enumerate(zip(trajectories, loss_functions, optimizers)):

        surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s).flatten()

        if optimizer is not None:
            surface_Z = optimizer(surface_Z.unsqueeze(1)).clone().detach().reshape(surface_X.size()).flatten()

        if loss_function is not None:
            surface_Z = loss_function.network(torch.cat((torch.cat((
                surface_X.reshape(-1, 1), surface_Y.reshape(-1, 1)), dim=1),
                surface_Z.unsqueeze(1)
            ), dim=1))

        # Reshaping and converting into the correct type.
        surface_Z = surface_Z.reshape(surface_X.size()).detach().numpy()

        # Plotting the contour plot, i.e. 2d loss landscape that we are optimizing over.
        axes[i].contourf(surface_X, surface_Y, surface_Z, levels=50, cmap="GnBu", zorder=0)

        x_trajectory = enforce_input_range([row[0] for row in trajectory])
        y_trajectory = enforce_input_range([row[1] for row in trajectory])

        axes[i].plot(x_trajectory, y_trajectory, c=path_colors[i], linewidth=3, zorder=10)
        axes[i].plot(x_trajectory[0], y_trajectory[0], c=path_colors[i], marker="o", markersize=8, zorder=20)
        axes[i].plot(x_trajectory[-1], y_trajectory[-1], c=path_colors[i], marker="*", markersize=15, zorder=20)

        # Clear the background of figure artifacts.
        clear_background(axes[i])

    if save:  # Saving the file to the given path.
        plt.savefig(file_name + ".pdf")
    else:  # Showing the figure.
        plt.show()

    plt.close()


def plot_landscape_2d_single(func, a, b, s, trajectories, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from X and Y values

    # Compute the Z values (function values)
    surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)

    # Plotting the contour plot, i.e. 2d loss landscape that we are optimizing over.
    axes.contourf(surface_X, surface_Y, surface_Z, levels=50, cmap="GnBu", zorder=0)

    for trajectory, color in zip(trajectories, path_colors):
        x_trajectory = enforce_input_range([row[0] for row in trajectory])
        y_trajectory = enforce_input_range([row[1] for row in trajectory])

        axes.plot(x_trajectory, y_trajectory, c=color, linewidth=3, zorder=10)
        axes.plot(x_trajectory[0], y_trajectory[0], c=color, marker="o", markersize=8, zorder=20)
        axes.plot(x_trajectory[-1], y_trajectory[-1], c=color, marker="*", markersize=15, zorder=20)

    # Clear the background of figure artifacts.
    clear_background(axes)

    if save:  # Saving the file to the given path.
        plt.savefig(file_name + ".pdf")
    else:  # Showing the figure.
        plt.show()

    plt.close()


# ============================================================
# Functions for generating 2D and 3D animations.
# ============================================================


def animate_landscape_3d_single(func, a, b, s, trajectories, save=False, file_name=""):

    # Generating the 3D plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []

    # For each of the given trajectories generate a path.
    for trajectory in trajectories:
        surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        trajectories_x.append(enforce_input_range([row[0] for row in trajectory]))
        trajectories_y.append(enforce_input_range([row[1] for row in trajectory]))
        trajectories_z.append([row[2] for row in trajectory])

    def animate(i):
        ax.cla()  # Clear the current plot

        # Plotting the wireframe, i.e. loss landscape that we are optimizing over.
        ax.plot_wireframe(surface_X, surface_Y, surface_Z, linewidth=0.75, edgecolor="#7881c4",
                          rstride=10, cstride=10, zorder=0)

        # Plotting each of the given trajectories up to the given frame.
        for trajectory_x, trajectory_y, trajectory_z, color in \
                zip(trajectories_x, trajectories_y, trajectories_z, path_colors):

            ax.plot(trajectory_x[:i], trajectory_y[:i], trajectory_z[:i], c=color, linewidth=3, zorder=10)
            ax.plot(trajectory_x[0], trajectory_y[0], trajectory_z[0], c=color, marker="o", markersize=7, zorder=20)

        # Clear the background of figure artifacts.
        clear_background(ax)

    # Generating the animation and showing or saving it.
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=100)

    if save:  # Saving the file to the given path.
        ani.save(file_name + ".gif", writer="ffmpeg")
    else:  # Showing the figure.
        plt.show()

    plt.close()


def animate_landscape_3d_multi(func, a, b, s, trajectories, loss_functions, optimizers, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(
        figsize=(len(loss_functions) * 7, 7),
        nrows=1, ncols=len(loss_functions),
        subplot_kw={"projection": "3d"},
        gridspec_kw={'top': 1, 'bottom': 0, 'wspace': 0, 'hspace': 0}
    )

    # Colours to use for the surfaces and trajectories.
    surface_colors = ["#ad67ca", "#7881c4", "#78b3c4", "#82b1aa", "#6fbc8b"]
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []
    loss_surfaces = []

    # For each of the given trajectories generate a path.
    for trajectory, loss_function, optimizer in zip(trajectories, loss_functions, optimizers):

        surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s).flatten()

        if optimizer is not None:
            surface_Z = optimizer(surface_Z.unsqueeze(1)).clone().detach().reshape(surface_X.size()).flatten()

        if loss_function is not None:
            surface_Z = loss_function.network(torch.cat((torch.cat((
                surface_X.reshape(-1, 1), surface_Y.reshape(-1, 1)), dim=1),
                surface_Z.unsqueeze(1)
            ), dim=1))

        trajectories_x.append(enforce_input_range([row[0] for row in trajectory]))
        trajectories_y.append(enforce_input_range([row[1] for row in trajectory]))
        trajectories_z.append([row[2] for row in trajectory])
        loss_surfaces.append(surface_Z.reshape(surface_X.size()).detach().numpy())

    def animate(i):

        # Plotting each of the given trajectories up to the given frame.
        for index, (trajectory_x, trajectory_y, trajectory_z, surface_Z) in \
                enumerate(zip(trajectories_x, trajectories_y, trajectories_z, loss_surfaces)):

            axes[index].cla()  # Clear the current plot

            # Plotting the wireframe, i.e. loss landscape that we are optimizing over.
            axes[index].plot_wireframe(surface_X, surface_Y, surface_Z, linewidth=0.75,
                                       edgecolor=surface_colors[index], rstride=10, cstride=10, zorder=0)

            axes[index].plot(trajectory_x[:i], trajectory_y[:i], trajectory_z[:i],
                             c=path_colors[index], linewidth=3, zorder=10)

            axes[index].plot(trajectory_x[0], trajectory_y[0], trajectory_z[0],
                             c=path_colors[index], marker="o", markersize=7, zorder=20)

            # Clear the background of figure artifacts.
            clear_background(axes[index])

    # Generating the animation and showing or saving it.
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=100)

    if save:  # Saving the file to the given path.
        ani.save(file_name + ".gif", writer="ffmpeg")
    else:  # Showing the figure.
        plt.show()

    plt.close()


def animate_landscape_2d_single(func, a, b, s, trajectories, save=False, file_name=""):

    # Generating the 3D plot.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []

    # For each of the given trajectories generate a path.
    for trajectory in trajectories:
        surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        trajectories_x.append(enforce_input_range([row[0] for row in trajectory]))
        trajectories_y.append(enforce_input_range([row[1] for row in trajectory]))

    def animate(i):
        ax.cla()  # Clear the current plot

        # Plotting the contour plot, i.e. 2d loss landscape that we are optimizing over.
        ax.contourf(surface_X, surface_Y, surface_Z, levels=50, cmap="GnBu", zorder=0)

        # Plotting each of the given trajectories up to the given frame.
        for trajectory_x, trajectory_y, color in zip(trajectories_x, trajectories_y, path_colors):

            ax.plot(trajectory_x[:i], trajectory_y[:i], c=color, linewidth=3, zorder=10)
            ax.plot(trajectory_x[0], trajectory_y[0], c=color, marker="o", markersize=7, zorder=20)

        # Clear the background of figure artifacts.
        clear_background(ax)

    # Generating the animation and showing or saving it.
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=100)

    if save:  # Saving the file to the given path.
        ani.save(file_name + ".gif", writer="ffmpeg")
    else:  # Showing the figure.
        plt.show()

    plt.close()


def animate_landscape_2d_multi(func, a, b, s, trajectories, loss_functions, optimizers, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(figsize=(len(loss_functions) * 7, 6), nrows=1, ncols=len(loss_functions))

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353", "#3a3a3a", "#3a3a3a"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y, indexing="ij")  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []
    loss_surfaces = []

    # For each of the given trajectories generate a path.
    for trajectory, loss_function, optimizer in zip(trajectories, loss_functions, optimizers):

        surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s).flatten()

        if optimizer is not None:
            surface_Z = optimizer(surface_Z.unsqueeze(1)).clone().detach().reshape(surface_X.size()).flatten()

        if loss_function is not None:
            surface_Z = loss_function.network(torch.cat((torch.cat((
                surface_X.reshape(-1, 1), surface_Y.reshape(-1, 1)), dim=1),
                surface_Z.unsqueeze(1)
            ), dim=1))

        trajectories_x.append(enforce_input_range([row[0] for row in trajectory]))
        trajectories_y.append(enforce_input_range([row[1] for row in trajectory]))
        loss_surfaces.append(surface_Z.reshape(surface_X.size()).detach().numpy())

    def animate(i):

        # Plotting each of the given trajectories up to the given frame.
        for index, (trajectory_x, trajectory_y, surface_Z, color) in \
                enumerate(zip(trajectories_x, trajectories_y, loss_surfaces, path_colors)):

            axes[index].cla()  # Clear the current plot

            # Plotting the contour plot, i.e. 2d loss landscape that we are optimizing over.
            axes[index].contourf(surface_X, surface_Y, surface_Z, levels=50, cmap="GnBu", zorder=0)
            axes[index].plot(trajectory_x[:i], trajectory_y[:i], c=color, linewidth=3, zorder=10)
            axes[index].plot(trajectory_x[0], trajectory_y[0], c=color, marker="o", markersize=7, zorder=20)

            # Clear the background of figure artifacts.
            clear_background(axes[index])

    # Generating the animation and showing or saving it.
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=100)

    if save:  # Saving the file to the given path.
        ani.save(file_name + ".gif", writer="ffmpeg")
    else:  # Showing the figure.
        plt.show()

    plt.close()


# ============================================================
# Utility functions.
# ============================================================


def clear_background(ax):

    if isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
        # Remove axis ticks and labels
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]);

        # Remove axis labels
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('');

        # Setting the the background pane colour to white.
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        # Hide axis spines
        ax.xaxis.line.set_visible(False); ax.yaxis.line.set_visible(False); ax.zaxis.line.set_visible(False)

        # Remove ticks on the axis spines
        ax.xaxis.set_ticklabels([]); ax.yaxis.set_ticklabels([]); ax.zaxis.set_ticklabels([])
    else:
        # Remove axis ticks and labels
        ax.set_xticks([]); ax.set_yticks([]);

        # Remove axis labels
        ax.set_xlabel(''); ax.set_ylabel('');

    # Show only the wireframe and nothing else
    ax.set_frame_on(False)

    # Remove grid
    ax.grid(False)


def enforce_input_range(trajectory, upper=3, lower=-3):
    path = []
    for value in trajectory:
        if value > upper:
            value = upper
        if value < lower:
            value = lower
        path.append(value)
    return path


if __name__ == "__main__":
    main()
