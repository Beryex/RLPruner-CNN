import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from conf import settings


def get_dataloader(dataset: str, 
                   batch_size: int = settings.T_BATCH_SIZE, 
                   num_workers: int = settings.T_NUM_WORKERS, 
                   shuffle: bool =True, 
                   val_proportion: float = settings.D_VAL_PROPORTION,
                   pin_memory: bool = False):
    if dataset == 'mnist':
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

        mnist_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        val_size = int(len(mnist_train_dataset) * val_proportion)
        train_size = len(mnist_train_dataset) - val_size
        mnist_train_dataset, mnist_val_dataset = random_split(mnist_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        mnist_train_loader = DataLoader(mnist_train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
        if val_size == 0:
            mnist_val_loader = None
        else:
            mnist_val_loader = DataLoader(mnist_val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
        
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        mnist_test_loader = DataLoader(mnist_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
        
        return mnist_train_loader, mnist_val_loader, mnist_test_loader, in_channels, num_class
    elif dataset == 'cifar10':
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

        cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_size = int(len(cifar10_train_dataset) * val_proportion)
        train_size = len(cifar10_train_dataset) - val_size
        cifar10_train_dataset, cifar10_val_dataset = random_split(cifar10_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        cifar10_train_loader = DataLoader(cifar10_train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
        if val_size == 0:
            cifar10_val_loader = None
        else:
            cifar10_val_loader = DataLoader(cifar10_val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        cifar10_test_loader = DataLoader(cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    
        return cifar10_train_loader, cifar10_val_loader, cifar10_test_loader, in_channels, num_class
    elif dataset == 'cifar100':
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

        cifar100_train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_size = int(len(cifar100_train_dataset) * val_proportion)
        train_size = len(cifar100_train_dataset) - val_size
        cifar100_train_dataset, cifar100_val_dataset = random_split(cifar100_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        cifar100_train_loader = DataLoader(cifar100_train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
        if val_size == 0:
            cifar100_val_loader = None
        else:
            cifar100_val_loader = DataLoader(cifar100_val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

        return cifar100_train_loader, cifar100_val_loader, cifar100_test_loader, in_channels, num_class
    else:
        return None, None, None, None, None
