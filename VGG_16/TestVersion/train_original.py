import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

from conf import settings
from utils import get_CIFAR10_training_dataloader, get_CIFAR10_test_dataloader, get_CIFAR100_training_dataloader, get_CIFAR100_test_dataloader, WarmUpLR
from models.vgg import VGG

def train(epoch):
    net.train()
    with tqdm(total=len(cifar10_training_loader), desc=f'Epoch {epoch}/{settings.ORIGINAL_EPOCH}', unit='img') as pbar:
        for batch_index, (images, labels) in enumerate(cifar10_training_loader):

            labels = labels.to(device)
            images = images.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch <= warm:
                warmup_scheduler.step()

            pbar.update(1)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

@torch.inference_mode()
def eval_training(epoch=0, tb=True):
    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    for (images, labels) in tqdm(cifar10_test_loader, total=len(cifar10_test_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    acc = correct.float() / len(cifar10_test_loader.dataset)
    logging.info('Top1 Accuracy: {}'.format(acc))
    return acc

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    logging.info('Start with random seed: {}'.format(current_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.1
    gamma = 0.2
    current_lr = lr
    warm = 1
    batch_size = 128

    #data preprocessing:
    cifar10_training_loader = get_CIFAR10_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    cifar10_test_loader = get_CIFAR10_test_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    net = VGG(num_class=10).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(cifar10_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    best_acc = 0.0

    for epoch in range(1, settings.ORIGINAL_EPOCH + 1):
        if epoch in settings.ORIGINAL_MILESTONES:
            current_lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.ORIGINAL_MILESTONES[1] and best_acc < acc:
            best_acc = acc
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, 'models/VGG_Original_{:d}.pkl'.format(current_time))
