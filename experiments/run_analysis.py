import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import datetime
import torch
import json

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    #plot_learning_curve()
    plot_pretraining_learning_curve()


def plot_learning_curve():

    paths = [
        "results/miniimagenet/pretrained-maml-miniimagenet-conv-5way-5shot-0.json",
        "results/miniimagenet/pretrained-maml-miniimagenet-conv-5way-5shot-1.json",
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)  # (8, 4.5)

    # Name of the methods and their respective plotting colors.
    method_names = ["Warp-noBMG", "Warp-BMG"]

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


def plot_pretraining_learning_curve():

    paths = [
        "results/miniimagenet/pretrained-maml-miniimagenet-conv-5way-1shot-0.json",
        "results/miniimagenet/pretrained-maml-miniimagenet-conv-5way-1shot-1.json",
        "results/miniimagenet/pretraining-maml-miniimagenet-conv-5way-1shot-3.json",
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)  # (8, 4.5)

    # Name of the methods and their respective plotting colors.
    method_names = ["0 - Proto Model", "1 - Final model", "2 - Proto Wide"]

    # Iterating over the different methods.
    for path, method in zip(paths, method_names):

        # Loading the json file into a dictionary.
        results = json.load(open(path))

        # Extracting the learning curve from the json file.
        loss = results["meta_training_history"]

        # Computing the average learning curve.
        plt.plot(np.linspace(0, len(loss), len(loss)), loss, linewidth=3, label=method)

        print(method, results["testing_mean"])

    plt.ylabel("Error")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
