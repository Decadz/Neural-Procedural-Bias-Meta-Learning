import torchvision
import torch
import numpy
import tarfile
import PIL
import os

# Finding the current directory of this file.
directory = os.path.dirname(os.path.realpath(__file__)) + "/datasets/"


def Omniglot(num_ways, num_shots, test_shots, device, pretraining=False, **kwargs):

    """
    Omniglot dataset which consists of 1,623 different handwritten characters from
    50 different alphabets, with only a few examples available for each character.
    The partitioning used comes from Vinyals et al. (2016).

    :param num_ways: Number of classes.
    :param num_shots: Number of training (support) instances.
    :param test_shots: Number of testing (query) instances.
    :param device: Device to put data on {"cpu", "cuda", ...}
    :param pretraining: If pretraining use a different set of transforms.
    :return: training, validation and testing datasets.
    """

    # Defining the training transforms.
    training_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(28),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

    # Defining the validation and testing transforms.
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(28),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

    # Generating the training, validation, and testing datasets.
    training = _Dataset("omniglot", set_name="train", transforms=training_transforms)
    validation = _Dataset("omniglot", set_name="val", transforms=testing_transforms)
    testing = _Dataset("omniglot", set_name="test", transforms=testing_transforms)

    # Generating the meta-learning task samplers.
    training_sampler = _TaskSampler(training, num_ways, num_shots, test_shots)
    validation_sampler = _TaskSampler(validation, num_ways, num_shots, test_shots)
    testing_sampler = _TaskSampler(testing, num_ways, num_shots, test_shots)

    # Generating a custom dataloaders which process batches.
    train = _DataLoader(training, training_sampler, device, random_rotation=True)
    val = _DataLoader(validation, validation_sampler, device, random_rotation=False)
    test = _DataLoader(testing, testing_sampler, device, random_rotation=False)

    return train, val, test


def CIFARFS(num_ways, num_shots, test_shots, device, pretraining=False, **kwargs):

    """
    The CIFAR-FS dataset uses a similar sampling procedure to miniImageNet
    (Ravi & Larochelle, 2017), CIFAR-FS is derived by randomly sampling 100
    classes from the 100 base classes in CIFAR100.

    :param num_ways: Number of classes.
    :param num_shots: Number of training (support) instances.
    :param test_shots: Number of testing (query) instances.
    :param device: Device to put data on {"cpu", "cuda", ...}
    :param pretraining: If pretraining use a different set of transforms.
    :return: training, validation and testing datasets.
    """

    # Defining the training transforms when not performing pretraining (i.e. meta learning).
    if not pretraining:
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    else:  # Defining the transforms when performing pretraining.
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    # Defining the validation and testing transforms.
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Generating the training, validation, and testing datasets.
    training = _Dataset("cifarfs", set_name="train", transforms=training_transforms)
    validation = _Dataset("cifarfs", set_name="val", transforms=testing_transforms)
    testing = _Dataset("cifarfs", set_name="test", transforms=testing_transforms)

    # Generating the meta-learning task samplers.
    training_sampler = _TaskSampler(training, num_ways, num_shots, test_shots)
    validation_sampler = _TaskSampler(validation, num_ways, num_shots, test_shots)
    testing_sampler = _TaskSampler(testing, num_ways, num_shots, test_shots)

    # Generating a custom dataloaders which process batches.
    train = _DataLoader(training, training_sampler, device, random_rotation=False)
    val = _DataLoader(validation, validation_sampler, device, random_rotation=False)
    test = _DataLoader(testing, testing_sampler, device, random_rotation=False)

    return train, val, test


def FC100(num_ways, num_shots, test_shots, device, pretraining=False, **kwargs):

    """
    The FC100 dataset (Fewshot-CIFAR100) is a new dataset proposed by Oreshkin et
    al. (2018) based on CIFAR-100 for few-shot learning. There are 60, 20, 20 classes
    in the training, validation, and testing sets, containing 600 images each.

    :param num_ways: Number of classes.
    :param num_shots: Number of training (support) instances.
    :param test_shots: Number of testing (query) instances.
    :param device: Device to put data on {"cpu", "cuda", ...}
    :param pretraining: If pretraining use a different set of transforms.
    :return: training, validation and testing datasets.
    """
    
    # Defining the training transforms when not performing pretraining (i.e. meta learning).
    if not pretraining:  
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    else:  # Defining the transforms when performing pretraining.
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    # Defining the validation and testing transforms.
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Generating the training, validation, and testing datasets.
    training = _Dataset("fc100", set_name="train", transforms=training_transforms)
    validation = _Dataset("fc100", set_name="val", transforms=testing_transforms)
    testing = _Dataset("fc100", set_name="test", transforms=testing_transforms)

    # Generating the meta-learning task samplers.
    training_sampler = _TaskSampler(training, num_ways, num_shots, test_shots)
    validation_sampler = _TaskSampler(validation, num_ways, num_shots, test_shots)
    testing_sampler = _TaskSampler(testing, num_ways, num_shots, test_shots)

    # Generating a custom dataloaders which process batches.
    train = _DataLoader(training, training_sampler, device, random_rotation=False)
    val = _DataLoader(validation, validation_sampler, device, random_rotation=False)
    test = _DataLoader(testing, testing_sampler, device, random_rotation=False)

    return train, val, test


def MiniImagenet(num_ways, num_shots, test_shots, device, pretraining=False, **kwargs):

    """
    The MiniImageNet dataset is a widely used benchmark dataset in few-shot learning.
    It is derived from the larger ImageNet dataset and consists of 100 classes with
    600 images per class.

    :param num_ways: Number of classes.
    :param num_shots: Number of training (support) instances.
    :param test_shots: Number of testing (query) instances.
    :param device: Device to put data on {"cpu", "cuda", ...}
    :param pretraining: If pretraining use a different set of transforms.
    :return: training, validation and testing datasets.
    """

    # Defining the training transforms when not performing pretraining (i.e. meta learning).
    if not pretraining:  
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(84),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:  # Defining the transforms when performing pretraining.
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(84),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Defining the validation and testing transforms.
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(84),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Generating the training, validation, and testing datasets.
    training = _Dataset("miniimagenet", set_name="train", transforms=training_transforms)
    validation = _Dataset("miniimagenet", set_name="val", transforms=testing_transforms)
    testing = _Dataset("miniimagenet", set_name="test", transforms=testing_transforms)

    # Generating the meta-learning task samplers.
    training_sampler = _TaskSampler(training, num_ways, num_shots, test_shots)
    validation_sampler = _TaskSampler(validation, num_ways, num_shots, test_shots)
    testing_sampler = _TaskSampler(testing, num_ways, num_shots, test_shots)

    # Generating a custom dataloaders which process batches.
    train = _DataLoader(training, training_sampler, device, random_rotation=False)
    val = _DataLoader(validation, validation_sampler, device, random_rotation=False)
    test = _DataLoader(testing, testing_sampler, device, random_rotation=False)

    return train, val, test


def TieredImagenet(num_ways, num_shots, test_shots, device, pretraining=False, **kwargs):

    """
    The tieredImageNet dataset is a larger subset of ILSVRC-12 proposed by Ren et al. (2018) with
    608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated
    hierarchy. This set of nodes is partitioned into 20, 6, and 8 disjoint sets of training,
    validation, and testing nodes, and the corresponding classes form the respective meta-sets.

    :param num_ways: Number of classes.
    :param num_shots: Number of training (support) instances.
    :param test_shots: Number of testing (query) instances.
    :param device: Device to put data on {"cpu", "cuda", ...}
    :param pretraining: If pretraining use a different set of transforms.
    :return: training, validation and testing datasets.
    """

    # Defining the training transforms when not performing pretraining (i.e. meta learning).
    if not pretraining:  
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(84),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:  # Defining the transforms when performing pretraining.
        training_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(84),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Defining the validation and testing transforms.
    testing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(84),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Generating the training, validation, and testing datasets.
    training = _Dataset("tieredimagenet", set_name="train", transforms=training_transforms)
    validation = _Dataset("tieredimagenet", set_name="val", transforms=testing_transforms)
    testing = _Dataset("tieredimagenet", set_name="test", transforms=testing_transforms)

    # Generating the meta-learning task samplers.
    training_sampler = _TaskSampler(training, num_ways, num_shots, test_shots)
    validation_sampler = _TaskSampler(validation, num_ways, num_shots, test_shots)
    testing_sampler = _TaskSampler(testing, num_ways, num_shots, test_shots)

    # Generating a custom dataloaders which process batches.
    train = _DataLoader(training, training_sampler, device, random_rotation=False)
    val = _DataLoader(validation, validation_sampler, device, random_rotation=False)
    test = _DataLoader(testing, testing_sampler, device, random_rotation=False)

    return train, val, test


class _Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, set_name, transforms):

        """
        A generic class for representing a PyTorch dataset, which loads data from either a .tar
        or a set of uncompressed folders (representing train, val, testing).

        :param dataset_name: Name of the dataset (folder).
        :param set_name: Dataset type = {train, val, test}.
        :param transforms: Torchvision transforms to apply to images.
        """

        # Extracting and replacing the .tar file if not already extracted.
        files = os.listdir(directory + dataset_name)
        if set_name not in files and set_name + ".tar" in files:
            with tarfile.open(directory + dataset_name + "/" + set_name + ".tar", 'r') as tar:
                tar.extractall(directory + dataset_name)
            os.remove(directory + dataset_name + "/" + set_name + ".tar")

        # Set the path according to train, val and test
        path = os.path.join(directory + dataset_name, set_name)
        classes = os.listdir(path)

        # Obtaining all the paths for each class.
        folders = []
        for label in classes:
            if os.path.isdir(os.path.join(path, label)):
                folders.append(os.path.join(path, label))

        # Generate empty list for data and label
        X, y = [], []

        # Get the images' paths and labels
        for idx, folder in enumerate(folders):
            instances = os.listdir(folder)
            for instance in instances:
                X.append(os.path.join(folder, instance))
                y.append(idx)

        # Defining the inputs outputs, and the relevant transforms.
        self.X, self.y = X, y
        self.transforms = transforms
        self.num_classes = len(classes)

    def __getitem__(self, i):
        path, label = self.X[i], self.y[i]
        image = PIL.Image.open(path, mode="r").convert("RGB")
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.X)


class _TaskSampler:

    def __init__(self, dataset, num_ways, num_shots, test_shots):

        """
        A task sampler object which is used for generating batches of tasks used
        for a standard multi-task meta-learning setup common in few-shot learning.

        :param dataset: PyTorch dataset object.
        :param num_ways: Number of classes (FSL ways).
        :param num_shots: Number of training (support) instances.
        :param test_shots: Number of testing (query) instances.
        """

        # Few-Shot Learning Settings.
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # List of tensors, where each tensor contains all labels from a given class.
        self.grouped_classes = []
        classes = numpy.array(dataset.y)

        # Sorting all instances into their respective groupings.
        for i in range(max(classes) + 1):
            group = numpy.argwhere(classes == i).reshape(-1)
            self.grouped_classes.append(torch.from_numpy(group))

    def __iter__(self):

        # Samples n_way random classes from the full dataset.
        classes = torch.randperm(len(self.grouped_classes))[:self.num_ways]

        batch = []
        for c in classes:  # For each class that is being sampled.
            single_class = self.grouped_classes[c]  # Extracting the tensor corresponding to the class.
            indexes = torch.randperm(len(single_class))[:self.num_shots + self.test_shots]
            batch.append(single_class[indexes])  # Appending the support and query to the batch.

        # Reshaping into the correct shape. Don't change otherwise classes will be mixed!
        yield torch.stack(batch).t().reshape(-1)


class _DataLoader:

    def __init__(self, dataset, sampler, device, random_rotation=False):

        """
        A dataloader object which wraps the dataset and sampler classes, and generates
        batches and assigns on the fly temporary class labels based on the dynamic batching.

        :param dataset: _Dataset object.
        :param sampler: _TaskSampler object.
        :param device: Device to put data on {"cpu", "cuda", ...}
        :param random_rotation: Whether to apply random rotations in {90, 180, 270}.
        """

        self.loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler)
        self.random_rotation = random_rotation
        self.dataset = dataset
        self.sampler = sampler
        self.device = device

    def __next__(self):

        # Generating a batch, and its corresponding labels, then sending to correct device.
        X, indices = next(iter(self.loader))
        y = torch.arange(self.sampler.num_ways).repeat(self.sampler.num_shots + self.sampler.test_shots)
        X, y = X.to(self.device), y.to(self.device)

        if self.random_rotation:  # Dynamically applying a random rotation to the ith class.
            for i in range(self.sampler.num_ways):
                random_rotation = numpy.random.choice([0, 1, 2, 3])
                X[i::self.sampler.num_ways] = torch.rot90(
                    X[i::self.sampler.num_ways], k=random_rotation, dims=[2, 3]
                )

        # Partitioning the batch into support and query sets.
        bs = self.sampler.num_ways * self.sampler.num_shots  # Computing the base batch size.
        X_support, y_support, X_query, y_query = X[:bs], y[:bs], X[bs:], y[bs:]

        # Generating the sorted indices for the support and query sets.
        sorted_support_indices = torch.argsort(y_support)
        sorted_query_indices = torch.argsort(y_query)

        # Sorting the indices such that the classes are now grouped.
        X_support, y_support = X_support[sorted_support_indices], y_support[sorted_support_indices]
        X_query, y_query = X_query[sorted_query_indices], y_query[sorted_query_indices]

        return X_support, y_support, X_query, y_query
