# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import copy
import math
from models.vgg import VGG
import models.vgg
from thop import profile


# global hyperparameter
train_num = 5               # how many training we are going to take
generate_num = 3            # for each updates, how many potential architecture we are going to generate
dev_num = 20                # for each potential architecture, how many epochs we are going to train it
accuracy_threshold = 0.7    # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #
max_tolerance_times = 3     # for each training, how many updates we are going to apply before we get the final architecture


def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct_1 = 0.0
    correct_5 = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        #compute top1
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        #compute top 5
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top1 Accuracy: {:.4f}, Top5 Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct_1 / len(cifar100_test_loader.dataset),
        correct_5 / len(cifar100_test_loader.dataset),
        finish - start
    ))

    return correct_1 / len(cifar100_test_loader.dataset), correct_5 / len(cifar100_test_loader.dataset)


def generate_architecture(model, local_top1_accuracy, local_top5_accuracy, generate_num, dev_num):
    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_function = nn.CrossEntropyLoss()

    # initialize all evaluating variables
    model_list = []
    top1_accuracy_list = []
    top5_accuracy_list = []
    FLOPs_list = []
    parameter_num_list = []
    model_list.append(model)
    top1_accuracy_list.append(local_top1_accuracy)
    top5_accuracy_list.append(local_top5_accuracy)
    input = torch.rand(128, 3, 32, 32).to(device)
    local_FLOPs, local_parameter_num = profile(model, inputs = (input, ), verbose=False)
    FLOPs_list.append(local_FLOPs)
    parameter_num_list.append(local_parameter_num)

    original_model = copy.deepcopy(model)
    for model_id in range(generate_num):
        # generate architecture
        dev_model = copy.deepcopy(original_model)
        VGG.update_architecture(dev_model)
        dev_model = dev_model.to(device)
        dev_lr = args.lr
        dev_optimizer = optim.SGD(dev_model.parameters(), lr=dev_lr, momentum=0.9, weight_decay=5e-4)
        dev_top1_accuracies = []
        dev_top5_accuracies = []
        # train the architecture for dev_num times
        for dev_id in range(1, dev_num + 1):
            if dev_id == 6 or dev_id == 13 or dev_id == 18:
                dev_lr *= gamma
                for param_group in dev_optimizer.param_groups:
                    param_group['lr'] = dev_lr
                print(dev_lr)
            # begin training
            dev_model.train()               # set model into training
            for idx, (train_x, train_label) in enumerate(cifar100_training_loader):
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
            
            # initialize the testing parameters
            correct_1 = 0.0
            correct_5 = 0.0

            # begin testing
            dev_model.eval()
            for idx, (test_x, test_label) in enumerate(cifar100_test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
                outputs = dev_model(test_x)
                _, preds = outputs.topk(5, 1, largest=True, sorted=True)
                #compute top1
                correct_1 += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
                #compute top 5
                top5_correct = test_label.view(-1, 1).expand_as(preds) == preds
                correct_5 += top5_correct.any(dim=1).sum().item()
            # calculate the accuracy and print it
            top1_accuracy = correct_1 / len(cifar100_test_loader.dataset)
            top5_accuracy = correct_5 / len(cifar100_test_loader.dataset)
            # discard the first half data as model need retraining
            if (dev_id + 1) % dev_num >= math.ceil(dev_num / 2):
                dev_top1_accuracies.append(top1_accuracy)
                dev_top5_accuracies.append(top5_accuracy)
        # store the model and score
        model_list.append(dev_model)
        top1_accuracy_list.append(dev_top1_accuracies)
        top5_accuracy_list.append(dev_top5_accuracies)
        input = torch.rand(128, 3, 32, 32).to(device)
        dev_FLOPs, dev_parameter_num = profile(dev_model, inputs = (input, ), verbose=False)
        FLOPs_list.append(dev_FLOPs)
        parameter_num_list.append(dev_parameter_num)
    score_list = compute_score(model_list, top1_accuracy_list, top5_accuracy_list, FLOPs_list, parameter_num_list)
    best_model_index = np.argmax(score_list)
    model = copy.deepcopy(model_list[best_model_index])
    print("model %d wins" %best_model_index)
    print(model.features)
    print(model.classifier)
    print(model.features['Conv6'].weight.shape)
    print(model.classifier[3].weight.shape)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    current_lr = args.lr

    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    print('Start with random seed %d' %current_time)
    tolerance_times = max_tolerance_times

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        gamma = 0.2
        if epoch in settings.MILESTONES:
            current_lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(current_lr)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        top1_acc, top5_acc = eval_training(epoch)

        # dynamic generate architecture
        if epoch >= 100 and epoch % 10 == 0:
            if tolerance_times > 0:
                net, model_index = copy.deepcopy(generate_architecture(net, [top1_acc], [top5_acc], generate_num, dev_num))
                if model_index == 0:
                    tolerance_times -= 1
                    models.vgg.max_modification_num -= 100
                    generate_num += 1
<<<<<<< HEAD:VGG_16/TestVersion/train_compressed.py
                    # models.vgg.kernel_neuron_proportion *= 0.6
=======
                    models.vgg.kernel_neuron_proportion *= 0.6
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:VGG_16/DynamicVersion/train.py
                optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
                print(current_lr)
            else:
                # save the module
<<<<<<< HEAD:VGG_16/TestVersion/train_compressed.py
                if not os.path.isdir("../models"):
                    os.mkdir("../models")
                torch.save(net, '../models/VGG_Compressed_{:d}.pkl'.format(current_time))
=======
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, 'models/vgg_{:d}.pkl'.format(current_time))
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:VGG_16/DynamicVersion/train.py
                break


        # save the model when training end
        if epoch == settings.EPOCH:
            # save the module
<<<<<<< HEAD:VGG_16/TestVersion/train_compressed.py
            if not os.path.isdir("../models"):
                os.mkdir("../models")
            torch.save(net, '../models/VGG_Compressed_{:d}.pkl'.format(current_time))
=======
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, 'models/vgg_{:d}.pkl'.format(current_time))
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:VGG_16/DynamicVersion/train.py
