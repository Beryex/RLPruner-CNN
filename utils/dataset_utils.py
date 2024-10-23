import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from typing import Tuple, Dict
import numpy as np
import torch

from conf import settings
from utils import torch_set_random_seed


DATASETS = ["cifar10", "cifar100"]


def get_dataloader(dataset_name: str, 
                   batch_size: int, 
                   num_workers: int,
                   calibration_num: int = 1000,
                   pin_memory: bool = True,
                   shuffle: bool =True) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """ Load and split dataset, also return dataset info for model building """
    # this function use independent random seed because it is more manageable for controling
    # reproduciable when resuming checkpoint
    torch_set_random_seed(1)
    if dataset_name == 'cifar10':
        num_classes = 10
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
        cifar10_train_loader = DataLoader(cifar10_train_dataset,
                                          shuffle=shuffle, 
                                          num_workers=num_workers, 
                                          batch_size=batch_size, 
                                          pin_memory=pin_memory,
                                          worker_init_fn=worker_init_fn)
        if calibration_num == 0:
            cifar10_calibrate_loader = None
        else:
            random_indices = np.random.choice(len(cifar10_train_dataset), calibration_num, replace=False)
            calibration_sampler = SubsetRandomSampler(random_indices)
            cifar10_calibrate_loader = DataLoader(cifar10_train_dataset, 
                                                  sampler=calibration_sampler, 
                                                  batch_size=calibration_num)

        cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                            train=False, 
                                                            download=True, 
                                                            transform=transform_test)
        cifar10_test_loader = DataLoader(cifar10_test_dataset, 
                                         shuffle=False, 
                                         num_workers=num_workers, 
                                         batch_size=batch_size, 
                                         pin_memory=pin_memory,
                                         worker_init_fn=worker_init_fn)
    
        return cifar10_train_loader, cifar10_calibrate_loader, cifar10_test_loader, num_classes
    
    elif dataset_name == 'cifar100':
        num_classes = 100
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
        cifar100_train_loader = DataLoader(cifar100_train_dataset, 
                                           shuffle=shuffle, 
                                           num_workers=num_workers, 
                                           batch_size=batch_size, 
                                           pin_memory=pin_memory,
                                           worker_init_fn=worker_init_fn)
        if calibration_num == 0:
            cifar100_calibrate_loader = None
        else:
            random_indices = np.random.choice(len(cifar100_train_dataset), calibration_num, replace=False)
            calibration_sampler = SubsetRandomSampler(random_indices)
            cifar100_calibrate_loader = DataLoader(cifar100_train_dataset, 
                                                   sampler=calibration_sampler, 
                                                   batch_size=calibration_num)

        cifar100_test_dataset = torchvision.datasets.CIFAR100(root='./data', 
                                                              train=False, 
                                                              download=True, 
                                                              transform=transform_test)
        cifar100_test_loader = DataLoader(cifar100_test_dataset, 
                                          shuffle=False, 
                                          num_workers=num_workers, 
                                          batch_size=batch_size, 
                                          pin_memory=pin_memory,
                                          worker_init_fn=worker_init_fn)

        return cifar100_train_loader, cifar100_calibrate_loader, cifar100_test_loader, num_classes
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def worker_init_fn(worker_id):
    np.random.seed(1 + worker_id)


def get_dataloader_with_checkpoint(prev_checkpoint: Dict,
                                   batch_size: int, 
                                   num_workers: int,
                                   pin_memory: bool = True,
                                   shuffle: bool =True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Resume the dataloader from the checkpoint """
    train_dataset = prev_checkpoint['train_loader']
    valid_dataset = prev_checkpoint['calibration_loader']
    test_dataset = prev_checkpoint['test_loader']
    
    train_sampler = prev_checkpoint['train_sampler']
    valid_sampler = prev_checkpoint['calibration_sampler']
    test_sampler = prev_checkpoint['test_sampler']
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=(train_sampler is None and shuffle), 
                              sampler=train_sampler, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory)
    calibration_loader = DataLoader(valid_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    sampler=valid_sampler, 
                                    num_workers=num_workers, 
                                    pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,
                             shuffle=False, 
                             sampler=test_sampler, 
                             num_workers=num_workers, 
                             pin_memory=pin_memory)
    
    return train_loader, calibration_loader, test_loader
