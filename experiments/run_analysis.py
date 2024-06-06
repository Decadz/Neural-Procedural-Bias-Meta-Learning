import matplotlib.pyplot as plt
import numpy as np
import torch
import json

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    plot_learning_curve()
    plot_pretraining_learning_curve()


def plot_learning_curve():

    paths = [
        "results/miniimagenet/npbml-miniimagenet-adaconv128-5way-5shot-0.json",
        "results/miniimagenet/npbml-miniimagenet-adaconv128-5way-1shot-0.json",

        "results/miniimagenet/npbml-miniimagenet-adaresnet-5way-5shot-0.json",
        "results/miniimagenet/npbml-miniimagenet-adaresnet-5way-1shot-0.json",

        "results/tieredimagenet/npbml-tieredimagenet-adaconv128-5way-5shot-0.json",
        "results/tieredimagenet/npbml-tieredimagenet-adaconv128-5way-1shot-0.json",

        "results/tieredimagenet/npbml-tieredimagenet-adaresnet-5way-5shot-0.json",
        "results/tieredimagenet/npbml-tieredimagenet-adaresnet-5way-1shot-0.json",
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)  # (8, 4.5)

    # Name of the methods and their respective plotting colors.
    method_names = ["1", "2", "3", "4", "5", "6", "7", "8"]

    # Iterating over the different methods.
    for path, method in zip(paths, method_names):

        # Loading the json file into a dictionary.
        results = json.load(open(path))

        # Extracting the learning curve from the json file.
        loss = results["meta_training_history"]

        # Computing the average learning curve.
        plt.plot(np.linspace(0, len(loss), len(loss)), loss, linewidth=3, label=method)
        print("Training:", method, "=", str(round(results["training_mean"], 5)))
        print("Testing:", method, "=", str(round(results["testing_mean"], 5)))
        print()

    plt.ylabel("Error")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_pretraining_learning_curve():

    paths = [
        "results/miniimagenet/pretraining-miniimagenet-adaconv128-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adaresnet-5way.json",

        "results/tieredimagenet/pretraining-tieredimagenet-adaconv128-5way.json",
        "results/tieredimagenet/pretraining-tieredimagenet-adaresnet-5way.json",
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)  # (8, 4.5)

    # Name of the methods and their respective plotting colors.
    method_names = ["1", "2", "3", "4"]

    # Iterating over the different methods.
    for path, method in zip(paths, method_names):

        # Loading the json file into a dictionary.
        results = json.load(open(path))

        # Extracting the learning curve from the json file.
        loss = results["meta_history"]  # meta_history, fine_tuning_history

        # Computing the average learning curve.
        plt.plot(np.linspace(0, len(loss), len(loss)), loss, linewidth=3, label=method)

        print(method, min(loss))  # results["testing_mean"]

    plt.ylabel("Error")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
