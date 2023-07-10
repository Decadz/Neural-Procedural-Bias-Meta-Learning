import torchmeta
import zipfile
import shutil
import json
import os

path = "../datasets/"

# Loading the relevant files into the project.
torchmeta.datasets.helpers.omniglot(
    path, ways=5, shots=5, test_shots=15,
    class_augmentations=None, target_transform=None, transform=None,
    meta_train=True, download=True, seed=0
)
torchmeta.datasets.helpers.omniglot(
    path, ways=5, shots=5, test_shots=15,
    class_augmentations=None, target_transform=None, transform=None,
    meta_val=True, download=True, seed=0
)
torchmeta.datasets.helpers.omniglot(
    path, ways=5, shots=5, test_shots=15,
    class_augmentations=None, target_transform=None, transform=None,
    meta_test=True, download=True, seed=0
)

print("Finished downloading relevant Omniglot files.")

path += "omniglot/"

# Unpacking the background images zip file.
with zipfile.ZipFile(path + "images_background.zip", 'r') as zip_ref:
    zip_ref.extractall(path)

# Unpacking the evaluation images zip file.
with zipfile.ZipFile(path + "images_evaluation.zip", 'r') as zip_ref:
    zip_ref.extractall(path)

# Generating the training, validation, and testing partitions.
for set_type in ["train", "val", "test"]:

    # Opening the json files which specify the vinyals split.
    with open(path + "vinyals_" + set_type + "_labels.json") as file:
        classes = json.load(file)

    # For each image class in
    for label in classes:

        # Defining the sources and destination folders.
        source_folder = path + label[0] + "/" + label[1] + "/" + label[2] + "/"
        destination_folder = path + set_type + "/" + label[1] + "_" + label[2] + "/"

        # If the folder already exists then we can skip, else will get permission errors.
        if os.path.exists(destination_folder) and os.path.isdir(destination_folder):
            continue

        # Copying the folder and all its images/items.
        shutil.copytree(source_folder, destination_folder)

    print("Finished generating", set_type, "set")
