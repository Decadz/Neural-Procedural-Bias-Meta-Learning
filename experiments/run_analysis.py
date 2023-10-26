import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import torch
import json

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    #compute_results()
    plot_learning_curve()


def compute_results():

    paths = [
        "results/mnist/online-mnist-lenet5",
        "results/mnist/quadratic-tp-mnist-lenet5",
        "results/mnist/cubic-tp-mnist-lenet5",
    ]

    # Iterating over the different methods.
    for path in paths:

        training, testing = [], []
        print(path)

        # Iterating over the random seeds/executions.
        for i in range(10, 20):

            # Loading the json file into a dictionary.
            res = json.load(open(path + "-" + str(i) + ".json"))

            # Extracting the training and testing inference from the results.
            training.append(np.mean(res["training_inference"]))
            testing.append(np.mean(res["testing_inference"]))
            #print("Seed:", str(i), res["testing_inference"])

        # Computing the mean +- std of the inference performance.
        training_mean, training_std = np.mean(training), np.std(training)
        testing_mean, testing_std = np.mean(testing), np.std(testing)

        # Displaying the results to the console.
        print("Training:", round(training_mean, 4), "$\pm$", round(training_std, 4))
        print("Testing:", round(testing_mean, 4), "$\pm$", round(testing_std, 4))
        print()


def plot_learning_curve():

    paths = [
        #"results/miniimagenet/warpgrad-miniimagenet-linearwarpconv-5way-1shot",

        "results/miniimagenet/warpgrad-miniimagenet-warpconv-5way-1shot-12.json",
        "results/miniimagenet/warpgrad-miniimagenet-warpconv-5way-1shot-13.json"
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)  # (8, 4.5)

    # Name of the methods and their respective plotting colors.
    method_names = ["CA-0.01", "CA-0.05"]

    # Iterating over the different methods.
    for path, method in zip(paths, method_names):

        # Loading the json file into a dictionary.
        results = json.load(open(path))

        # Extracting the learning curve from the json file.
        loss = results["meta_training_history"]

        # Computing the average learning curve.
        plt.plot(np.linspace(0, len(loss), len(loss)), loss, linewidth=3, label=method)
        print(method, "=", str(round(results["testing_mean"], 5)))

    plt.ylabel("Error")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # plt.savefig("meta-training-svhn-wideresnet.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
