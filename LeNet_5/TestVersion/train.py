from model import Model
import numpy as np
import os
import torch
import visdom
import time
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def train_paperversion():
    global cur_batch_window
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_train = MNIST('./data/mnist',
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=1024, shuffle=True, num_workers=8)
    all_epoch = 200
    train_time = 1
    for train_id in range(train_time):
        # reinitialize random seed
        current_time = int(time.time())
        torch.manual_seed(current_time)
        print('Start with random seed %d' %current_time)

        model = Model().to(device)
        sgd = SGD(model.parameters(), lr=1e-1)
        loss_fn = CrossEntropyLoss()
        prev_acc = 0
        for current_epoch in range(all_epoch):
            model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                sgd.zero_grad()
                predict_y = model(train_x.float())
                loss = loss_fn(predict_y, train_label.long())
                loss.backward()
                sgd.step()

            # initialize the testing parameters
            top1_correct_num = 0.0
            top3_correct_num = 0.0

            model.eval()
            
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                outputs = model(test_x)
                _, preds = outputs.topk(3, 1, largest=True, sorted=True)
                top1_correct_num += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
                top3_correct = test_label.view(-1, 1).expand_as(preds) == preds
                top3_correct_num += top3_correct.any(dim=1).sum().item()
            # calculate the accuracy and print it
            top1_accuracy = top1_correct_num / len(test_loader.dataset)
            top3_accuracy = top3_correct_num / len(test_loader.dataset)
            print('top1 accuracy: {:.3f}'.format(top1_accuracy), flush=True)
            print('top3 accuracy: {:.3f}'.format(top3_accuracy), flush=True)
            if np.abs(top1_accuracy - prev_acc) < 1e-4:
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(model, 'models/mnist_{:.3f}.pkl'.format(current_time))
                break
            prev_acc = top1_accuracy


if __name__ == '__main__':
    train_paperversion()
    print("Model finished training")