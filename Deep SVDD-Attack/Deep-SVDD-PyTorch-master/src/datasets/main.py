from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .svhn import SVHN_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10','svhn','cifar100')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'svhn':
        dataset = SVHN_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'cifar100':
        dataset = CIFAR100_Dataset(root=data_path, normal_class=normal_class)

    return dataset
