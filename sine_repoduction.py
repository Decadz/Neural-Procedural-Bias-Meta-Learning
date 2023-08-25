import matplotlib.pyplot as plt
import torch
import higher
import numpy


def main():

    # Setting the reproducibility seeds.
    torch.cuda.manual_seed_all(2000)
    torch.cuda.manual_seed(2000)
    torch.manual_seed(2000)
    numpy.random.seed(2000)

    # Creating a meta learned loss function.
    learned_loss_function = LearnedLossFunction()

    # initialize task loss for meta training
    task_loss_fn = TaskLoss()

    # Performing the meta-training process.
    learned_loss_function = meta_train(learned_loss_function, task_loss_fn)
    plot_loss_landscape(learned_loss_function)


class TaskLoss(object):
    def __call__(self, y_hat, y, v_hat, v, shaped):
        if shaped:
            loss = (v_hat - v) ** 2
        else:
            loss = (y_hat - y) ** 2
        return loss


class BaseModel(torch.nn.Module):

    def __init__(self, theta=None):
        super(BaseModel, self).__init__()
        if theta is None:
            self.freq = torch.nn.Parameter(torch.Tensor([0.1]))  # TODO - Test out other default values.
        else:
            self.freq = torch.nn.Parameter(torch.Tensor([theta]))

    def forward(self, x):
        return torch.sin(self.freq*x)


class LearnedLossFunction(torch.nn.Module):

    def __init__(self):
        super(LearnedLossFunction, self).__init__()
        self.loss_fn = torch.nn.Sequential(
            torch.nn.Linear(3, 10), torch.nn.ELU(),
            torch.nn.Linear(10, 10), torch.nn.ELU(),
            torch.nn.Linear(10, 10), torch.nn.ELU(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.loss_fn(x)


def sine_dataset(num_tasks, num_examples_task, num_steps, freq_range=[-5.0, 5.0], input_range=[-5.0, 5.0]):
    """ Generate samples from random sine functions. """
    freq = numpy.random.uniform(freq_range[0], freq_range[1], [num_tasks])  # List containing one frequency per task.

    # Initializing the the lists containing the dataset.
    init_inputs = numpy.zeros([num_tasks, num_steps, num_examples_task, 1])  # Input (x) information (4, 10, 64, 1)
    outputs = numpy.zeros([num_tasks, num_steps, num_examples_task, 1])  # Label (y) information (4, 10, 64, 1)
    thetas = numpy.zeros([num_tasks, num_steps, num_examples_task, 1])  # Frequrncy in correct shape (4, 10, 64, 1)

    # For each task populate the arrays.
    for task in range(num_tasks):
        init_inputs[task] = numpy.repeat(
            numpy.random.uniform(input_range[0], input_range[1], [1, num_examples_task, 1]), num_steps, -3)

        outputs[task] = numpy.sin(freq[task] * init_inputs[task])
        thetas[task] = numpy.zeros_like(outputs[task]) + freq[task]

    return init_inputs, outputs, thetas


def meta_train(learned_loss, task_loss_fn, n_outer_iter=1500, shaped=True, num_task=4, n_inner_iter=10):

    # Creating a meta optimizer for the learned loss function.
    meta_opt = torch.optim.Adam(learned_loss.parameters(), lr=0.001)

    # Performing the outer optimization loop.
    for outer_i in range(n_outer_iter):

        # set gradient with respect to meta loss parameters to 0
        batch_inputs, batch_labels, batch_thetas = sine_dataset(
            num_tasks=num_task, num_examples_task=64, num_steps=n_inner_iter
        )

        # For each task in the task distribution p(T).
        for task in range(num_task):

            sine_model = BaseModel()  # Creating the base model f_{\theta}(x).
            inner_opt = torch.optim.SGD([sine_model.freq], lr=1.0)  # Base optimizer.

            # For each inner optimization step.
            for step in range(n_inner_iter):

                inputs = torch.Tensor(batch_inputs[task, step, :])  # Sampling the inputs (x)
                labels = torch.Tensor(batch_labels[task, step, :])  # Sampling the labels (y)
                label_thetas = torch.Tensor(batch_thetas[task, step, :])  # Sampling the target frequency (v).

                with higher.innerloop_ctx(sine_model, inner_opt) as (fmodel, diffopt):

                    # use current meta loss to update model
                    yp = fmodel(inputs)                                  # TODO Does x actually help as a meta input?
                    meta_input = torch.cat([inputs, yp, labels], dim=1)  # Given x, y_hat and y.

                    meta_out = learned_loss(meta_input)   # Feeding values to meta-learned loss function
                    loss = meta_out.mean()            # Computing the reduction.
                    diffopt.step(loss)                # Updating the base model weights.

                    yp = fmodel(inputs)   # Computing another forward pass.
                    # Custom meta objective or just normal MSE loss.
                    task_loss = task_loss_fn(yp, labels, fmodel.freq, label_thetas, shaped)

                # Updating the learned loss function after every
                meta_opt.zero_grad()
                task_loss.mean().backward()
                meta_opt.step()

                sine_model.freq = torch.nn.Parameter(fmodel.freq.clone().detach())  # Remove
                inner_opt = torch.optim.SGD([sine_model.freq], lr=1.0)  # Remove

        if outer_i % 100 == 0:
            print(outer_i, "task loss:", task_loss.mean().item())

    return learned_loss


def plot_loss_function(sine_model, ml3_loss, opt_iter, test_x, test_y):
    opt = torch.optim.SGD(sine_model.parameters(), lr=0.1)
    for i in range(opt_iter):
        yp = sine_model(test_x)
        meta_input = torch.cat([test_x, yp, test_y], dim=1)
        pred_task_loss = ml3_loss(meta_input).mean()
        opt.zero_grad()
        pred_task_loss.backward()
        opt.step()
        yp = sine_model(test_x)
        plt.plot(yp.detach().numpy())
        plt.plot(test_y.detach().numpy())
        plt.show()


def plot_loss_landscape(meta_loss_function, freq=0.5):

    theta_ranges = numpy.arange(-7.0, 7.0, 0.1)  # Sampling a grid between [-7, 7]
    test_x = numpy.expand_dims(numpy.arange(-5.0, 5.0, 0.1), 1)
    test_y = numpy.sin(freq * test_x)
    x, y = torch.Tensor(test_x), torch.Tensor(test_y)

    loss_landscape = []
    for theta in theta_ranges:  # Sampling the performance at all the different weight combinations.
        base_model = BaseModel(theta)
        y_hat = base_model(x)
        loss = 0.5 * (y_hat - y) ** 2
        loss_landscape.append(loss.mean().detach().numpy())

    plt.plot(theta_ranges, loss_landscape, c='C1')
    plt.xlabel('Theta')
    plt.ylabel('Loss')
    plt.legend(['MSE Loss Landscape'])
    plt.axvline(x=freq, c='red')
    plt.show()
    plt.gca()

    meta_loss_landscape = []
    for theta in theta_ranges:  # Sampling the performance at all the different weight combinations.
        base_model = BaseModel(theta)
        y_hat = base_model(x)
        meta_input = torch.cat([x, y_hat, y], 1)
        loss = meta_loss_function(meta_input).mean()
        meta_loss_landscape.append(loss.clone().mean().detach().numpy())

    plt.plot(theta_ranges, meta_loss_landscape)
    plt.ylabel('Loss')
    plt.legend(['Shaped ML$^3$ Landscape'])
    plt.axvline(x=freq, c='red')
    plt.show()


if __name__ == "__main__":
    main()
