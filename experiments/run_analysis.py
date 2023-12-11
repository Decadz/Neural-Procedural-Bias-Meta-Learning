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
        #"results/miniimagenet/npbml-miniimagenet-adaconv32-5way-1shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv48-5way-1shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv64-5way-1shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv128-5way-1shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv32-5way-5shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv48-5way-5shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv64-5way-5shot-0.json",
        #"results/miniimagenet/npbml-miniimagenet-adaconv128-5way-5shot-0.json"
        "results/miniimagenet/npbml-ablating-meta-learning-rate/npbml-miniimagenet-adawarpconv32-5way-5shot-4.json",
        "results/miniimagenet/npbml-ablating-meta-learning-rate/npbml-miniimagenet-adawarpconv32-5way-5shot-5.json",
        "results/miniimagenet/npbml-ablating-meta-learning-rate/npbml-miniimagenet-adawarpconv32-5way-5shot-6.json",
        "results/miniimagenet/npbml-ablating-meta-learning-rate/npbml-miniimagenet-adawarpconv32-5way-5shot-7.json",
        "results/miniimagenet/npbml-ablating-meta-learning-rate/npbml-miniimagenet-adawarpconv32-5way-5shot-8.json",
        "results/miniimagenet/npbml-ablating-meta-learning-rate/npbml-miniimagenet-adawarpconv32-5way-5shot-9.json",
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)  # (8, 4.5)

    # Name of the methods and their respective plotting colors.
    method_names = ["Cosine 1e-3 -> 1e-5", "Cosine 1e-4 -> 1e5", "MS 1e-3 - [20k 40k] g=0.1", "MS 1e-3 - [10k, 20k, 30k, 40k, 50k] g=0.5", "MS 1e-4 -> [30k] g=0.1", "MS 1e-4 -> [10k, 20k, 30k, 40k, 50k] g=0.7"]

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


def plot_pretraining_learning_curve():

    paths = [
        "results/miniimagenet/pretraining-miniimagenet-adaconv32-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adaconv48-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adaconv64-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adaconv128-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adawarpconv32-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adawarpconv48-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adawarpconv64-5way.json",
        "results/miniimagenet/pretraining-miniimagenet-adawarpconv128-5way.json",

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
        loss = results["meta_history"]  # meta_history, fine_tuning_history

        # Computing the average learning curve.
        plt.plot(np.linspace(0, len(loss), len(loss)), loss, linewidth=3, label=method)

        #print(method, results["testing_mean"])

    plt.ylabel("Error")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
