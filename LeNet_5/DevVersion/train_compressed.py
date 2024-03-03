<<<<<<< HEAD:LeNet_5/DevVersion/train_compressed.py
from models.lenet import LeNet
=======
from lenet import LeNet
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:LeNet_5/DynamicVersion/train.py
import numpy as np
import os
import torch
import visdom
import math
import copy
import time
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from thop import profile
<<<<<<< HEAD:LeNet_5/DevVersion/train_compressed.py
import models.lenet as lenet
=======
import lenet
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:LeNet_5/DynamicVersion/train.py

# global hyperparameter
train_num = 1               # how many training we are going to take
generate_num = 5            # for each updates, how many potential architecture we are going to generate
dev_num = 15                # for each potential architecture, how many epochs we are going to train it
accuracy_threshold = 0.95   # if current top1 accuracy is above the accuracy_threshold, then computation of architecture's score main focus on FLOPs and parameter #
max_tolerance_times = 7        # for each training, how many updates we are going to apply before we get the final architecture

# load train and test data
batch_size = 256
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

# move the LeNet Module into the corresponding device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_dynamic_architecture():
    # initialize the training parameters
    max_train_epoches_num = 200
    global generate_num

    for train_id in range(train_num):
        # reinitialize random seed
        current_time = int(time.time())
        current_time = 1708791769
        torch.manual_seed(current_time)
        print('Start with random seed %d' %current_time)

        train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=4)
        test_loader = DataLoader(data_test, batch_size=1024, shuffle=True, num_workers=4)
        prev_accuracy = 0
        tolerance_times = max_tolerance_times
        # reload the model each time
        model = LeNet().to(device)
        sgd = SGD(model.parameters(), lr = 1e-1)
        loss_function = CrossEntropyLoss()
        for current_epoch_num in range(max_train_epoches_num):
            
            # begin training
            model.train()               # set model into training
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # clear the gradient data and update parameters based on error
                sgd.zero_grad()
                # get predict y and compute the error
                predict_y = model(train_x)
                loss = loss_function(predict_y, train_label)
                loss.backward()
                sgd.step()
            
            # initialize the testing parameters
            top1_correct_num = 0.0
            top3_correct_num = 0.0

            # begin testing
            model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
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
    
            # if reach all epochs
            if current_epoch_num == max_train_epoches_num - 1:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
<<<<<<< HEAD:LeNet_5/DevVersion/train_compressed.py
                torch.save(model, 'models/LeNet_Compressed_{:d}.pkl'.format(current_time))
=======
                torch.save(model, 'models/mnist_{:d}.pkl'.format(current_time))
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:LeNet_5/DynamicVersion/train.py
                print("Single training have %f top1 accuracy using model with architexcture [%d, %d, %d, %d, %s, %s, %s, %s, %s]" %(top1_accuracy, model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features, model.fc2.out_features, str(model.conv1_activation_func), str(model.conv2_activation_func), str(model.fc1_activation_func), str(model.fc2_activation_func), str(model.fc3_activation_func)))
            
            # check and update architecture
            if np.abs(top1_accuracy - prev_accuracy) < 1e-4:
                if tolerance_times > 0:
                    # update architecture
                    # find the potential best architecture\
                    model, model_index = copy.deepcopy(generate_architecture(model, [top1_accuracy], [top3_accuracy], generate_num, dev_num))
<<<<<<< HEAD:LeNet_5/DevVersion/train_compressed.py
                    model = model.to(device)
=======
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:LeNet_5/DynamicVersion/train.py
                    sgd = SGD(model.parameters(), lr = 1e-1)
                    if model_index == 0:
                        tolerance_times -= 1
                        lenet.max_modification_num -= 5
                        generate_num += 1
                    prev_accuracy = 0
                    continue
                else:
                    # save the module
                    if not os.path.isdir("models"):
                        os.mkdir("models")
<<<<<<< HEAD:LeNet_5/DevVersion/train_compressed.py
                    torch.save(model, 'models/LeNet_Compressed_{:d}.pkl'.format(current_time))
=======
                    torch.save(model, 'models/mnist_{:d}.pkl'.format(train_id))
>>>>>>> 0dd8a320783e68dd14a5e93ce12bc78cdaf71e58:LeNet_5/DynamicVersion/train.py
                    print("Single training have %f top1 accuracy using model with architexcture [%d, %d, %d, %d, %s, %s, %s, %s, %s]" %(top1_accuracy, model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features, model.fc2.out_features, str(model.conv1_activation_func), str(model.conv2_activation_func), str(model.fc1_activation_func), str(model.fc2_activation_func), str(model.fc3_activation_func)))
                    break
            prev_accuracy = top1_accuracy


def generate_architecture(model, local_top1_accuracy, local_top3_accuracy, generate_num, dev_num):
    loss_function = CrossEntropyLoss()

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
        train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
        test_loader = DataLoader(data_test, batch_size=1024, shuffle=True, num_workers=8)
        # generate architecture
        dev_model = copy.deepcopy(original_model)
        LeNet.update_architecture(dev_model)
        dev_model = dev_model.to(device)
        sgd = SGD(dev_model.parameters(), lr = 1e-1)
        dev_top1_accuracies = []
        dev_top3_accuracies = []
        print("Train potential model with [%d, %d, %d, %d, %s, %s, %s, %s, %s]" %(dev_model.conv1.out_channels, dev_model.conv2.out_channels, dev_model.fc1.out_features, dev_model.fc2.out_features, str(dev_model.conv1_activation_func), str(dev_model.conv2_activation_func), str(dev_model.fc1_activation_func), str(dev_model.fc2_activation_func), str(dev_model.fc3_activation_func)))
        # train the architecture for dev_num times
        for dev_id in range(dev_num):
            # begin training
            dev_model.train()               # set model into training
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # clear the gradient data and update parameters based on error
                sgd.zero_grad()
                # get predict y and compute the error
                predict_y = dev_model(train_x.float())
                loss = loss_function(predict_y, train_label.long())
                loss.backward()
                sgd.step()
            
            # initialize the testing parameters
            top1_correct_num = 0.0
            top3_correct_num = 0.0

            # begin testing
            dev_model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
                outputs = dev_model(test_x)
                _, preds = outputs.topk(3, 1, largest=True, sorted=True)
                top1_correct_num += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
                top3_correct = test_label.view(-1, 1).expand_as(preds) == preds
                top3_correct_num += top3_correct.any(dim=1).sum().item()
            # calculate the accuracy and print it
            top1_accuracy = top1_correct_num / len(test_loader.dataset)
            top3_accuracy = top3_correct_num / len(test_loader.dataset)
            # discard the first half data as model need retraining
            if (dev_id + 1) % dev_num >= math.ceil(dev_num / 2):
                dev_top1_accuracies.append(top1_accuracy)
                dev_top3_accuracies.append(top3_accuracy)
        # store the model and score
        model_list.append(dev_model)
        top1_accuracy_list.append(dev_top1_accuracies)
        top3_accuracy_list.append(dev_top3_accuracies)
        input = torch.rand(256, 1, 32, 32).to(device)
        dev_FLOPs, dev_parameter_num = profile(dev_model, inputs = (input, ), verbose=False)
        FLOPs_list.append(dev_FLOPs)
        parameter_num_list.append(dev_parameter_num)
    score_list = compute_score(model_list, top1_accuracy_list, top3_accuracy_list, FLOPs_list, parameter_num_list)
    best_model_index = np.argmax(score_list)
    model = copy.deepcopy(model_list[best_model_index])
    print("model %d wins with %d conv1_kernel_num and %d conv2_kernel_num" %(best_model_index, model.conv1.out_channels, model.conv2.out_channels))
    print(model.conv2.weight.shape)
    print(model.fc2.weight.shape)
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
    train_dynamic_architecture()
    print("Model finished training")