import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import numpy as np

from conf import settings
from utils import torch_set_random_seed


DATASETS = ["mnist", "cifar10", "cifar100"]


def get_dataloader(dataset_name: str, 
                   batch_size: int, 
                   num_workers: int,
                   val_proportion: float = 0,
                   pin_memory: bool = True,
                   shuffle: bool =True) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """ Load and split dataset, also return dataset info for model building """
    # this function use independent random seed because it is more manageable for controling
    # reproduciable when resuming checkpoint
    torch_set_random_seed(1)
    if dataset_name == 'mnist':
        in_channels = 1
        num_class = 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.D_MNIST_TRAIN_MEAN, settings.D_MNIST_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(settings.D_MNIST_TRAIN_MEAN, settings.D_MNIST_TRAIN_STD)
        ])

        mnist_train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                         train=True, 
                                                         download=True, 
                                                         transform=transform_train)
        val_size = int(len(mnist_train_dataset) * val_proportion)
        train_size = len(mnist_train_dataset) - val_size
        mnist_train_dataset, mnist_val_dataset = random_split(mnist_train_dataset, 
                                                              [train_size, val_size])
        mnist_train_loader = DataLoader(mnist_train_dataset, 
                                        shuffle=shuffle, 
                                        num_workers=num_workers, 
                                        batch_size=batch_size, 
                                        pin_memory=pin_memory,
                                        worker_init_fn=worker_init_fn)
        if val_size == 0:
            mnist_val_loader = None
        else:
            mnist_val_loader = DataLoader(mnist_val_dataset, 
                                          shuffle=shuffle, 
                                          num_workers=num_workers, 
                                          batch_size=batch_size, 
                                          pin_memory=pin_memory,
                                          worker_init_fn=worker_init_fn)
        
        mnist_test_dataset = torchvision.datasets.MNIST(root='./data', 
                                                        train=False, 
                                                        download=True, 
                                                        transform=transform_test)
        mnist_test_loader = DataLoader(mnist_test_dataset, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers, 
                                       batch_size=batch_size, 
                                       pin_memory=pin_memory,
                                       worker_init_fn=worker_init_fn)
        
        return mnist_train_loader, mnist_val_loader, mnist_test_loader, in_channels, num_class
    
    elif dataset_name == 'cifar10':
        in_channels = 3
        num_class = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.D_CIFAR10_TRAIN_MEAN, settings.D_CIFAR10_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.D_CIFAR10_TRAIN_MEAN, settings.D_CIFAR10_TRAIN_STD)
        ])

        cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                             train=True, 
                                                             download=True, 
                                                             transform=transform_train)
        val_size = int(len(cifar10_train_dataset) * val_proportion)
        train_size = len(cifar10_train_dataset) - val_size
        cifar10_train_dataset, cifar10_val_dataset = random_split(cifar10_train_dataset, 
                                                                  [train_size, val_size])
        cifar10_train_loader = DataLoader(cifar10_train_dataset,
                                          shuffle=shuffle, 
                                          num_workers=num_workers, 
                                          batch_size=batch_size, 
                                          pin_memory=pin_memory,
                                          worker_init_fn=worker_init_fn)
        if val_size == 0:
            cifar10_val_loader = None
        else:
            cifar10_val_loader = DataLoader(cifar10_val_dataset, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers, 
                                            batch_size=batch_size, 
                                            pin_memory=pin_memory,
                                            worker_init_fn=worker_init_fn)

        cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                            train=False, 
                                                            download=True, 
                                                            transform=transform_test)
        cifar10_test_loader = DataLoader(cifar10_test_dataset, 
                                         shuffle=shuffle, 
                                         num_workers=num_workers, 
                                         batch_size=batch_size, 
                                         pin_memory=pin_memory,
                                         worker_init_fn=worker_init_fn)
    
        return cifar10_train_loader, cifar10_val_loader, cifar10_test_loader, in_channels, num_class
    
    elif dataset_name == 'cifar100':
        in_channels = 3
        num_class = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.D_CIFAR100_TRAIN_MEAN, settings.D_CIFAR100_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.D_CIFAR100_TRAIN_MEAN, settings.D_CIFAR100_TRAIN_STD)
        ])

        cifar100_train_dataset = torchvision.datasets.CIFAR100(root='./data', 
                                                               train=True, 
                                                               download=True, 
                                                               transform=transform_train)
        val_size = int(len(cifar100_train_dataset) * val_proportion)
        train_size = len(cifar100_train_dataset) - val_size
        cifar100_train_dataset, cifar100_val_dataset = random_split(cifar100_train_dataset, 
                                                                    [train_size, val_size])
        cifar100_train_loader = DataLoader(cifar100_train_dataset, 
                                           shuffle=shuffle, 
                                           num_workers=num_workers, 
                                           batch_size=batch_size, 
                                           pin_memory=pin_memory,
                                           worker_init_fn=worker_init_fn)
        if val_size == 0:
            cifar100_val_loader = None
        else:
            cifar100_val_loader = DataLoader(cifar100_val_dataset, 
                                             shuffle=shuffle, 
                                             num_workers=num_workers, 
                                             batch_size=batch_size, 
                                             pin_memory=pin_memory,
                                             worker_init_fn=worker_init_fn)

        cifar100_test_dataset = torchvision.datasets.CIFAR100(root='./data', 
                                                              train=False, 
                                                              download=True, 
                                                              transform=transform_test)
        cifar100_test_loader = DataLoader(cifar100_test_dataset, 
                                          shuffle=shuffle, 
                                          num_workers=num_workers, 
                                          batch_size=batch_size, 
                                          pin_memory=pin_memory,
                                          worker_init_fn=worker_init_fn)

        return cifar100_train_loader, cifar100_val_loader, cifar100_test_loader, in_channels, num_class
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def worker_init_fn(worker_id):
    np.random.seed(1 + worker_id)
