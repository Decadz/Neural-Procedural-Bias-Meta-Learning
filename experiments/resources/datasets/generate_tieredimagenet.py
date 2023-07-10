import tarfile

# Download dataset from: https://lyy.mpi-inf.mpg.de/mtl/download/

# Path to the relevant "tiered-imagenet.tar" file.
directory = "../datasets/tieredimagenet/"
dataset_name = "tiered-imagenet.tar"

# Extracting the .tar file and creating a new directory.
with tarfile.open(directory + dataset_name, 'r') as tar:
    tar.extractall()
