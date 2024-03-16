import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from conf import settings
from utils import get_CIFAR10_training_dataloader, get_CIFAR10_test_dataloader, get_MNIST_training_dataloader, get_MNIST_test_dataloader, WarmUpLR

import copy
import math
from models.lenet import LeNet
from thop import profile


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
    correct_1 = 0.0
    correct_3 = 0.0

    for (images, labels) in mnist_test_loader:

        images = images.to(device)
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.topk(3, 1, largest=True, sorted=True)
        #compute top1
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        #compute top 5
        top3_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_3 += top3_correct.any(dim=1).sum().item()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top1 Accuracy: {:.4f}, Top5 Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(mnist_test_loader.dataset),
        correct_1 / len(mnist_test_loader.dataset),
        correct_3 / len(mnist_test_loader.dataset),
        finish - start
    ))

    return correct_1 / len(mnist_test_loader.dataset), correct_3 / len(mnist_test_loader.dataset)


def generate_architecture(model, local_top1_accuracy, local_top3_accuracy):
    loss_function = nn.CrossEntropyLoss()

    # initialize all evaluating variables
    model_list = []
    top1_accuracy_list = []
    top3_accuracy_list = []
    FLOPs_list = []
    parameter_num_list = []
    model_list.append(model)
    top1_accuracy_list.append(local_top1_accuracy)
    top3_accuracy_list.append(local_top3_accuracy)
    input = torch.rand(256, 1, 32, 32).to(device)
    local_FLOPs, local_parameter_num = profile(model, inputs = (input, ), verbose=False)
    FLOPs_list.append(local_FLOPs)
    parameter_num_list.append(local_parameter_num)

    original_model = copy.deepcopy(model)
    for model_id in range(generate_num):
        # generate architecture
        dev_model = copy.deepcopy(original_model)
        LeNet.update_architecture(dev_model, modification_num)
        dev_model = dev_model.to(device)
        dev_lr = lr
        dev_optimizer = optim.SGD(dev_model.parameters(), lr=dev_lr, momentum=0.9, weight_decay=5e-4)
        dev_top1_accuracies = []
        dev_top3_accuracies = []
        # train the architecture for dev_num times
        for dev_id in range(1, dev_num + 1):
            if dev_id in settings.DYNAMIC_MILESTONES:
                dev_lr *= gamma
                for param_group in dev_optimizer.param_groups:
                    param_group['lr'] = dev_lr
            # begin training
            dev_model.train()               # set model into training
            for train_x, train_label in mnist_training_loader:
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # clear the gradient data and update parameters based on error
                dev_optimizer.zero_grad()
                # get predict y and compute the error
                predict_y = dev_model(train_x)
                loss = loss_function(predict_y, train_label)
                # update visualization
                loss.backward()
                dev_optimizer.step()
            
            # discard the first half data as model need retraining
            if (dev_id + 1) % dev_num >= math.ceil(dev_num / 2):
                # initialize the testing parameters
                correct_1 = 0.0
                correct_3 = 0.0
                # begin testing
                dev_model.eval()
                with torch.no_grad():
                    for test_x, test_label in mnist_test_loader:
                        # move test data to device
                        test_x = test_x.to(device)
                        test_label = test_label.to(device)
                        # get predict y and predict its class
                        outputs = dev_model(test_x)
                        _, preds = outputs.topk(3, 1, largest=True, sorted=True)
                        #compute top1
                        correct_1 += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
                        #compute top 5
                        top3_correct = test_label.view(-1, 1).expand_as(preds) == preds
                        correct_3 += top3_correct.any(dim=1).sum().item()
                    # calculate the accuracy and print it
                    top1_accuracy = correct_1 / len(mnist_test_loader.dataset)
                    top3_accuracy = correct_3 / len(mnist_test_loader.dataset)
                    dev_top1_accuracies.append(top1_accuracy)
                    dev_top3_accuracies.append(top3_accuracy)
        # store the model and score
        model_list.append(dev_model)
        top1_accuracy_list.append(dev_top1_accuracies)
        top3_accuracy_list.append(dev_top3_accuracies)
        dev_FLOPs, dev_parameter_num = profile(dev_model, inputs = (input, ), verbose=False)
        FLOPs_list.append(dev_FLOPs)
        parameter_num_list.append(dev_parameter_num)
    score_list = compute_score(model_list, top1_accuracy_list, top3_accuracy_list, FLOPs_list, parameter_num_list)
    best_model_index = np.argmax(score_list)
    best_model_FLOPs = FLOPs_list[best_model_index]
    best_model_Params = parameter_num_list[best_model_index]
    FLOPs_compressed_ratio = best_model_FLOPs / local_FLOPs
    Para_compressed_ratio = best_model_Params / local_parameter_num
    model = copy.deepcopy(model_list[best_model_index])
    print("model %d wins" %best_model_index)
    print("This compression ratio: FLOPs: %f, Parameter number: %f" %(FLOPs_compressed_ratio, Para_compressed_ratio))
    return model, best_model_index


def compute_score(model_list, top1_accuracy_list, top3_accuracy_list, FLOPs_list, parameter_num_list):
    print(top1_accuracy_list)
    score_list = []
    # extract the last element (converged) accuracy to denote that architecture's accuracy
    top1_accuracies = [sublist[-1] for sublist in top1_accuracy_list]
    top3_accuracies = [sublist[-1] for sublist in top3_accuracy_list]
    # use Min-Max Normalization to process the FLOPs_list and parameter_num_list
    FLOPs_tensor = torch.tensor(FLOPs_list)
    parameter_num_tensor = torch.tensor(parameter_num_list)
    FLOPs_scaled = (FLOPs_tensor - torch.min(FLOPs_tensor)) / (torch.max(FLOPs_tensor) - torch.min(FLOPs_tensor))
    parameter_num_scaled = (parameter_num_tensor - torch.min(parameter_num_tensor)) / (torch.max(parameter_num_tensor) - torch.min(parameter_num_tensor))
    for model_id in range(len(model_list)):
        top1_accuracy = top1_accuracies[model_id]
        top3_accuracy = top3_accuracies[model_id]
        if np.max(top1_accuracies) > accuracy_threshold:
            # if there exists architecture that is higher than accuracy_threshold, only pick the simplest one and discard other
            if (top1_accuracy > accuracy_threshold - 0.005):
                score_list.append(top1_accuracy * 0.5 +  0.5 - FLOPs_scaled[model_id].item() * 0.25 - parameter_num_scaled[model_id].item() * 0.25)
            else:
                score_list.append(0)
        else:
            score_list.append(top1_accuracy * 0.9 +  top3_accuracy * 0.1)
    print(FLOPs_scaled)
    print(score_list)
    return score_list


if __name__ == '__main__':
    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.1
    gamma = 0.2
    current_lr = lr * gamma * gamma
    warm = 1
    generate_num = settings.MAX_GENERATE_NUM
    modification_num = settings.MAX_MODIFICATION_NUM
    tolerance_times = settings.MAX_TOLERANCE_TIMES
    dev_num = settings.DEV_NUM
    accuracy_threshold = settings.ACCURACY_THRESHOLD

    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    print('Start with random seed %d' %current_time)

    net = torch.load('models/LeNet_Original_1710607419.pkl')  # replace it with the model gained by train_original.py
    net = net.to(device)

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
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(mnist_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    for epoch in range(1, settings.DYNAMIC_EPOCH + 1):
        train(epoch)
        top1_acc, top5_acc = eval_training(epoch)

        # dynamic generate architecture
        if epoch % 10 == 0:
            if tolerance_times > 0:
                net, model_index = copy.deepcopy(generate_architecture(net, [top1_acc], [top5_acc]))
                net = net.to(device)
                if model_index == 0:
                    tolerance_times -= 1
                    modification_num /= 2
                    generate_num += 1
                optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, 'models/LeNet_Compressed_{:d}.pkl'.format(current_time))
            else:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, 'models/LeNet_Compressed_{:d}.pkl'.format(current_time))
                break


        # save the model when training end
        if epoch == settings.DYNAMIC_EPOCH:
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, 'models/LeNet_Compressed_{:d}.pkl'.format(current_time))
