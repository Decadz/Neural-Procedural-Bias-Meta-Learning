import torch
import os


def export_model(model, res_directory, file_name, save_state_dictionary=True, separate_models_directory=True):

    # Exporting the results to a json file.
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    # Determining whether to use a nested model directory.
    directory = res_directory + "models/" if separate_models_directory else res_directory

    if model is not None:  # Exporting the trained model to a .pth file.
        if not os.path.exists(directory):
            os.makedirs(directory)

        if save_state_dictionary:
            torch.save(model.state_dict(), directory + file_name + ".pth")
        else:
            torch.save(model, directory + file_name + ".pth")


def export_loss(loss, res_directory, file_name, save_state_dictionary=True):

    # Exporting the results to a json file.
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    if loss is not None:  # Exporting the trained model to a .pth file.
        if not os.path.exists(res_directory + "losses/"):
            os.makedirs(res_directory + "losses/")

        if save_state_dictionary:
            torch.save(loss.state_dict(), res_directory + "losses/" + file_name + ".pth")
        else:
            torch.save(loss, res_directory + "losses/" + file_name + ".pth")


def export_results(results, res_directory, file_name):

    # Exporting the results to a json file.
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    _export_file(results, res_directory + file_name + ".json")


def _export_file(results, file_name):

    """
    Takes a dictionary containing the results and experimental configurations
    and exports it to a .json file in the desired directory. If the directory
    does not exist it is created.

    :param results: Dictionary containing the experimental results.
    :param file_name: File name for the output results file.
    """

    # Removing administrative information.
    del results["experiment_configuration"]["output_path"]
    del results["experiment_configuration"]["verbose"]

    if "description" in results["experiment_configuration"]:
        # Converting description to a string.
        results["experiment_configuration"]["description"] = \
            ' '.join(results["experiment_configuration"]["description"])

    def _format_dictionary(dictionary, level=0, indent=4):
        string = ""
        if isinstance(dictionary, dict):
            string += "{" + "\n"
            comma = ""
            for key, value in dictionary.items():
                string += comma
                comma = ",\n"
                string += " " * indent * (level + 1)
                string += '"' + str(key) + '":' + " "
                string += _format_dictionary(value, level + 1)

            string += "\n" + " " * indent * level + "}"
        elif isinstance(dictionary, str):
            string += '"' + dictionary + '"'
        elif isinstance(dictionary, list):
            string += "[" + ",".join([_format_dictionary(e, level + 1) for e in dictionary]) + "]"
        elif isinstance(dictionary, tuple):
            string += "[" + ",".join(_format_dictionary(e, level + 1) for e in dictionary) + "]"
        elif isinstance(dictionary, bool):
            string += "true" if dictionary else "false"
        elif isinstance(dictionary, int):
            string += str(dictionary)
        elif isinstance(dictionary, float):
            string += "%.7g" % dictionary
        elif dictionary is None:
            string += "null"
        else:
            raise TypeError("Unknown type '%s' for json" % str(type(dictionary)))
        return string

    # Exporting all the results and information to a .json file.
    file = open(file_name, "w+")
    file.write(_format_dictionary(results))
    file.close()
