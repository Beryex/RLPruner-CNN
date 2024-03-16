import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from conf import settings
from utils import get_CIFAR10_training_dataloader, get_CIFAR10_test_dataloader, get_MNIST_training_dataloader, get_MNIST_test_dataloader, WarmUpLR

from models.lenet import LeNet

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(mnist_training_loader):

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= warm:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in mnist_test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(mnist_test_loader.dataset),
        correct.float() / len(mnist_test_loader.dataset),
        finish - start
    ))

    return correct.float() / len(mnist_test_loader.dataset)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.1
    gamma = 0.2
    current_lr = lr
    warm = 1
    batch_size = 128

    net = LeNet(num_class=10).to(device)

    #data preprocessing:
    mnist_training_loader = get_MNIST_training_dataloader(
        num_workers=4,
        batch_size=256,
        shuffle=True
    )

    mnist_test_loader = get_MNIST_test_dataloader(
        num_workers=4,
        batch_size=1024,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr)
    iter_per_epoch = len(mnist_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    print('Start with random seed %d' %current_time)

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
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, 'models/LeNet_Original_{:d}.pkl'.format(current_time))
            continue
