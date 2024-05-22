import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from conf import settings


def get_dataloader(dataset: str, 
                   batch_size: int = 16, 
                   num_workers: int = 2, 
                   shuffle: bool =True, 
                   dataset_proportion: float = 0.2):
    if dataset == 'mnist':
        in_channels = 1
        num_class = 10
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        mnist_training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        mnist_training_loader = DataLoader(mnist_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        n_prototyping = int(len(mnist_training) * dataset_proportion)
        n_discard = len(mnist_training) - n_prototyping
        mnist_prototyping, _ = random_split(mnist_training, [n_prototyping, n_discard], generator=torch.Generator().manual_seed(0))
        mnist_prototyping_loader = DataLoader(mnist_prototyping, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        mnist_test_loader = DataLoader(mnist_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        
        return mnist_training_loader, mnist_prototyping_loader, mnist_test_loader, in_channels, num_class
    elif dataset == 'cifar10':
        in_channels = 3
        num_class = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD)
        ])

        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        cifar10_training_loader = DataLoader(cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        n_prototyping = int(len(cifar10_training) * dataset_proportion)
        n_discard = len(cifar10_training) - n_prototyping
        cifar10_prototyping, _ = random_split(cifar10_training, [n_prototyping, n_discard], generator=torch.Generator().manual_seed(0))
        cifar10_prototyping_loader = DataLoader(cifar10_prototyping, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        cifar10_test_loader = DataLoader(cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
        return cifar10_training_loader, cifar10_prototyping_loader, cifar10_test_loader, in_channels, num_class
    elif dataset == 'cifar100':
        in_channels = 3
        num_class = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])

        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        n_prototyping = int(len(cifar100_training) * dataset_proportion)
        n_discard = len(cifar100_training) - n_prototyping
        cifar100_prototyping, _ = random_split(cifar100_training, [n_prototyping, n_discard], generator=torch.Generator().manual_seed(0))
        cifar100_prototyping_loader = DataLoader(cifar100_prototyping, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        return cifar100_training_loader, cifar100_prototyping_loader, cifar100_test_loader, in_channels, num_class
    else:
        return None, None, None, None, None
