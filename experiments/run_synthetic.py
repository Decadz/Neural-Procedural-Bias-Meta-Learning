import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits
import higher
import random
import torch
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
    meta_network_1 = learned_loss_function_1(func)
    meta_network_2 = learned_loss_function_2(func)

    # Performing the meta testing phase on the following seeds (selected because they look nice).
    for seed in [1, 2, 8, 10, 12, 14, 17, 23, 25, 33, 34, 42, 46]:

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

        npbml_trajectory_1 = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s, loss_fn=meta_network_1)

        npbml_trajectory_2 = gradient_descent(
            torch.tensor([x1, x2], requires_grad=True), func, a=a, b=b, s=s, loss_fn=meta_network_2)

        # Plotting each of the loss landscapes, and then the trajectories on those landscapes.
        paths = [sgd_trajectory["true"], npbml_trajectory_1["loss"], npbml_trajectory_2["loss"]]
        loss_fns = [None, meta_network_1, meta_network_2]
        plot_landscape_3d_multi(func, a, b, s, paths, loss_fns, save=False, file_name=path + "3d-multi-" + str(seed))
        plot_landscape_2d_multi(func, a, b, s, paths, loss_fns, save=False, file_name=path + "2d-multi-" + str(seed))
        animate_landscape_3d_multi(func, a, b, s, paths, loss_fns, save=False, file_name=path + "3d-multi-" + str(seed))
        animate_landscape_2d_multi(func, a, b, s, paths, loss_fns, save=False, file_name=path + "2d-multi-" + str(seed))

        # Plotting the true landscapes, and then the trajectories on that single landscapes.
        paths = [sgd_trajectory["true"], npbml_trajectory_1["true"], npbml_trajectory_2["true"]]
        plot_landscape_3d_single(func, a, b, s, paths, save=False, file_name=path + "3d-single-" + str(seed))
        plot_landscape_2d_single(func, a, b, s, paths, save=False, file_name=path + "2d-single-" + str(seed))
        animate_landscape_3d_single(func, a, b, s, paths, save=False, file_name=path + "3d-single-" + str(seed))
        animate_landscape_2d_single(func, a, b, s, paths, save=False, file_name=path + "2d-single-" + str(seed))


# ============================================================
# Training related code for the different methods.
# ============================================================


def func(x, a, b, s):
    return b[0] * (a[0] - x[0]) ** 2 * torch.exp(-x[0] ** 2 - (x[1] + a[1]) ** 2) \
           - b[1] * (x[0] / s - x[0] ** 3 - x[1] ** 5) * torch.exp(-x[0] ** 2 - x[1] ** 2) \
           - b[2] * torch.exp(-(x[0] + a[2]) ** 2 - x[0] ** 2)


def gradient_descent(x, func, a, b, s, loss_fn=None):

    optimizer = torch.optim.SGD([x], lr=0.1)  # lr is the learning rate
    true_trajectory, loss_trajectory = [], []

    for step in range(100):

        if loss_fn is None:
            loss = func(x, a=a, b=b, s=s)
            true_trajectory.append([x[0].item(), x[1].item(), loss.item()])
        else:
            y = func(x, a=a, b=b, s=s)
            loss = loss_fn(x, y, a, b, s)
            true_trajectory.append([x[0].item(), x[1].item(), y.item()])
            loss_trajectory.append([x[0].item(), x[1].item(), loss.item()])

        # Compute gradients
        loss.backward()

        # Update tensor values based on gradients
        optimizer.step()

        # Clear gradients for the next step
        optimizer.zero_grad()

    return {"true": true_trajectory, "loss": loss_trajectory}


class LossNetwork(torch.nn.Module):

    def __init__(self):
        super(LossNetwork, self).__init__()

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(10, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x, y, a, b, s):
        return self.network(torch.cat((x, y, a, b, s), dim=0))


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.s = torch.randint(1, 11, (1,))
        self.a = torch.randint(-1, 2, (3,))
        self.b = torch.randint(-5, 6, (3,))
        self.x = torch.nn.Parameter(torch.rand(2, requires_grad=True) * 6 - 3)

    def forward(self, func):
        return func(self.x, self.a, self.b, self.s)


def learned_loss_function_1(func):

    meta_network = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_optimizer = torch.optim.Adam(meta_network.parameters(), lr=0.001)

    # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
    for step in range(200):

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
                    base_loss = meta_network(fmodel.x, y, fmodel.a, fmodel.b, fmodel.s)
                    diffopt.step(base_loss)  # Update base network weights (theta).

                # Computing the task loss and updating the meta weights.
                y = fmodel(func)  # Finding the loss wrt. meta (task) loss.
                y.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        print("step", step, ":", y.item())

        # Update meta-loss network weights (phi).
        meta_optimizer.step()

    return meta_network


def learned_loss_function_2(func):

    meta_network = LossNetwork()

    # Defining the outer optimizer for the meta-loss network.
    meta_optimizer = torch.optim.Adam(meta_network.parameters(), lr=0.001)

    # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
    for step in range(200):

        # Clearing the gradient cache.
        meta_optimizer.zero_grad()

        # For each training task in the task distribution.
        for i in range(10):

            base_model = Model()
            base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

            # Approximating the ground truth minimum location x using random search
            rs = (torch.rand((2, 100), requires_grad=True) * 6 - 3).detach().requires_grad_(False)
            rs_results = func(rs, base_model.a, base_model.b, base_model.s)
            x_opt = torch.tensor([rs[0][torch.argmin(rs_results)], rs[1][torch.argmin(rs_results)]])

            # Taking a predetermined number of inner steps before meta update.
            for inner_steps in range(10):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_model, base_optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                    # Computing the loss using the learned loss and updating the base weights.
                    y = fmodel(func)
                    base_loss = meta_network(fmodel.x, y, fmodel.a, fmodel.b, fmodel.s)
                    diffopt.step(base_loss)  # Update base network weights (theta).

                # Computing the task loss and updating the meta weights.
                task_loss = ((x_opt - fmodel.x) ** 2).sum()
                task_loss.backward()  # Accumulates gradients wrt. to meta parameters.

                base_model.x = torch.nn.Parameter(fmodel.x.clone().detach())
                base_optimizer = torch.optim.SGD([base_model.x], lr=0.1)

        print("step", step, ":", y.item())

        # Update meta-loss network weights (phi).
        meta_optimizer.step()

    return meta_network


# ============================================================
# Functions for generating 2D and 3D static visualizations.
# ============================================================


def plot_landscape_3d_multi(func, a, b, s, trajectories, loss_functions, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(
        figsize=(len(trajectories) * 7, 7),
        nrows=1, ncols=len(trajectories),
        subplot_kw={"projection": "3d"},
        gridspec_kw={'top': 1, 'bottom': 0, 'wspace': 0, 'hspace': 0}
    )

    # Colours to use for the surfaces and trajectories.
    surface_colors = ["#7881c4", "#78b3c4", "#6fbc8b"]
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from X and Y values

    for i, (trajectory, loss_fn) in enumerate(zip(trajectories, loss_functions)):

        if loss_fn is None:
            surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        else:
            Y = func([surface_X, surface_Y], a, b, s)
            with torch.no_grad():
                Z = []
                for x1, x2, y in zip(surface_X.flatten(), surface_Y.flatten(), Y.flatten()):
                    Z.append(loss_fn(torch.tensor([x1, x2]), torch.tensor([y]), a, b, s))
            surface_Z = torch.tensor(Z).reshape(surface_X.shape)

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


def plot_landscape_3d_single(func, a, b, s, trajectories, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(figsize=(10, 8), nrows=1, ncols=1, subplot_kw={"projection": "3d"})

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from X and Y values

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


def plot_landscape_2d_multi(func, a, b, s, trajectories, loss_functions, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(
        figsize=(len(trajectories) * 7, 6),
        nrows=1, ncols=len(trajectories)
    )

    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from X and Y values

    for i, (trajectory, loss_fn) in enumerate(zip(trajectories, loss_functions)):

        if loss_fn is None:
            surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        else:
            Y = func([surface_X, surface_Y], a, b, s)
            with torch.no_grad():
                Z = []
                for x1, x2, y in zip(surface_X.flatten(), surface_Y.flatten(), Y.flatten()):
                    Z.append(loss_fn(torch.tensor([x1, x2]), torch.tensor([y]), a, b, s))
            surface_Z = torch.tensor(Z).reshape(surface_X.shape)

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


def plot_landscape_2d_single(func, a, b, s, trajectories, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values
    y = torch.linspace(-3, 3, 300)  # Y values
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from X and Y values

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


# ============================================================
# Functions for generating 2D and 3D animations.
# ============================================================


def animate_landscape_3d_single(func, a, b, s, trajectories, loss_function=None, save=False, file_name=""):

    # Generating the 3D plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []

    # For each of the given trajectories generate a path.
    for trajectory in trajectories:

        if loss_function is None:
            surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        else:
            fx = func([surface_X, surface_Y], a, b, s)
            with torch.no_grad():
                Z = []  # Generating the z position based on the position on the given loss function.
                for x1, x2, f in zip(surface_X.flatten(), surface_Y.flatten(), fx.flatten()):
                    Z.append(loss_function(torch.tensor([x1, x2]), torch.tensor([f]), a, b, s))
            surface_Z = torch.tensor(Z).reshape(surface_X.shape)

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


def animate_landscape_3d_multi(func, a, b, s, trajectories, loss_functions, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(
        figsize=(len(loss_functions) * 7, 7),
        nrows=1, ncols=len(loss_functions),
        subplot_kw={"projection": "3d"},
        gridspec_kw={'top': 1, 'bottom': 0, 'wspace': 0, 'hspace': 0}
    )

    # Colours to use for the surfaces and trajectories.
    surface_colors = ["#7881c4", "#78b3c4", "#6fbc8b"]
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []
    loss_surfaces = []

    # For each of the given trajectories generate a path.
    for trajectory, loss_function in zip(trajectories, loss_functions):

        if loss_function is None:
            surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        else:
            fx = func([surface_X, surface_Y], a, b, s)
            with torch.no_grad():
                Z = []  # Generating the z position based on the position on the given loss function.
                for x1, x2, f in zip(surface_X.flatten(), surface_Y.flatten(), fx.flatten()):
                    Z.append(loss_function(torch.tensor([x1, x2]), torch.tensor([f]), a, b, s))
            surface_Z = torch.tensor(Z).reshape(surface_X.shape)

        trajectories_x.append(enforce_input_range([row[0] for row in trajectory]))
        trajectories_y.append(enforce_input_range([row[1] for row in trajectory]))
        trajectories_z.append([row[2] for row in trajectory])
        loss_surfaces.append(surface_Z)

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


def animate_landscape_2d_single(func, a, b, s, trajectories, loss_function=None, save=False, file_name=""):

    # Generating the 3D plot.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []

    # For each of the given trajectories generate a path.
    for trajectory in trajectories:

        if loss_function is None:
            surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        else:
            fx = func([surface_X, surface_Y], a, b, s)
            with torch.no_grad():
                Z = []  # Generating the z position based on the position on the given loss function.
                for x1, x2, f in zip(surface_X.flatten(), surface_Y.flatten(), fx.flatten()):
                    Z.append(loss_function(torch.tensor([x1, x2]), torch.tensor([f]), a, b, s))
            surface_Z = torch.tensor(Z).reshape(surface_X.shape)

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


def animate_landscape_2d_multi(func, a, b, s, trajectories, loss_functions, save=False, file_name=""):

    # Generating the 3D plot.
    fig, axes = plt.subplots(figsize=(len(loss_functions) * 7, 6), nrows=1, ncols=len(loss_functions))

    # Colours to use for the trajectories.
    path_colors = ["#3a3a3a", "#924444", "#ce5353"]

    # Create a grid of points
    x = torch.linspace(-3, 3, 300)  # X values uniformly spaced.
    y = torch.linspace(-3, 3, 300)  # Y values uniformly spaced.
    surface_X, surface_Y = torch.meshgrid(x, y)  # Create a grid from the x and y values.

    # Creating lists which contain all the trajectories for the x, y and z coordinates.
    trajectories_x, trajectories_y, trajectories_z = [], [], []
    loss_surfaces = []

    # For each of the given trajectories generate a path.
    for trajectory, loss_function in zip(trajectories, loss_functions):

        if loss_function is None:
            surface_Z = func([surface_X, surface_Y], a=a, b=b, s=s)  # Compute the Z values (function values)
        else:
            fx = func([surface_X, surface_Y], a, b, s)
            with torch.no_grad():
                Z = []  # Generating the z position based on the position on the given loss function.
                for x1, x2, f in zip(surface_X.flatten(), surface_Y.flatten(), fx.flatten()):
                    Z.append(loss_function(torch.tensor([x1, x2]), torch.tensor([f]), a, b, s))
            surface_Z = torch.tensor(Z).reshape(surface_X.shape)

        trajectories_x.append(enforce_input_range([row[0] for row in trajectory]))
        trajectories_y.append(enforce_input_range([row[1] for row in trajectory]))
        loss_surfaces.append(surface_Z)

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


# ============================================================
# Utility functions.
# ============================================================


def clear_background(ax):

    if isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
        # Remove axis ticks and labels
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]);

        # Remove axis labels
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('');

        # Setting the background pane colour to white.
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
