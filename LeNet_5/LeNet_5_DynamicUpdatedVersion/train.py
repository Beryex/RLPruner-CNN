from model import LeNet
import numpy as np
import os
import torch
import visdom
import math
import copy
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
    all_epoch = 64
    train_time = 10
    final_accuracies_list = []
    train_list = []
    for train_id in range(train_time):
        train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
        test_loader = DataLoader(data_test, batch_size=1024, shuffle=True, num_workers=8)
        train_list.append(train_id)
        model = LeNet().to(device)
        sgd = SGD(model.parameters(), lr=2e-2)
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

            all_correct_num = 0
            all_sample_num = 0
            model.eval()
            
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                predict_y = model(test_x.float()).detach()
                predict_y =torch.argmax(predict_y, dim=-1)
                current_correct_num = predict_y == test_label
                all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                all_sample_num += current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            print('accuracy: {:.3f}'.format(acc), flush=True)
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
            if np.abs(acc - prev_acc) < 1e-4 and current_epoch >= 20:
                final_accuracies_list.append(acc)
                print("Single training have %f top1 accuracy" %acc)
                break
            if current_correct_num == all_epoch - 1:
                final_accuracies_list.append(acc)
                print("Single training have %f top1 accuracy" %acc)
            prev_acc = acc
    # visualization
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(final_accuracies_list), torch.Tensor(train_list),
                            win=cur_batch_window, name='Training ID %d' %train_id,
                            update=(None if cur_batch_window is None else 'append'),
                            opts=cur_batch_window_opts)


def train_dynamic_architecture():
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

    # initialize the training parameters
    max_train_epoches_num = 1000
    global cur_batch_window
    train_num = 1
    generate_num = 5
    dev_num = 15
    train_list = []
    final_accuracies_list = []

    for train_id in range(train_num):
        train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
        test_loader = DataLoader(data_test, batch_size=1024, shuffle=True, num_workers=8)
        train_list.append(train_id)
        prev_accuracy = 0
        update_times = 7
        # reload the model each time
        model = LeNet().to(device)
        sgd = SGD(model.parameters(), lr = 1e-1)
        loss_function = CrossEntropyLoss()
        local_top1_accuracy = []
        local_top3_accuracy = []
        epoch_list = []
        display_accuracies_list = []
        for current_epoch_num in range(max_train_epoches_num):
            epoch_list.append(current_epoch_num)
            
            # begin training
            model.train()               # set model into training
            loss_list, batch_list = [], []
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # clear the gradient data and update parameters based on error
                sgd.zero_grad()
                # get predict y and compute the error
                predict_y = model(train_x.float())
                loss = loss_function(predict_y, train_label.long())
                # update visualization
                loss_list.append(loss.detach().cpu().item())
                batch_list.append(idx + 1)
                loss.backward()
                sgd.step()
            
            # initialize the testing parameters
            top1_correct_num = 0
            top3_correct_num = 0
            test_sample_num = 0

            # begin testing
            model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
                predict_y = model(test_x.float()).detach()
                top1_correct_num += topk_correct_num(predict_y, test_label, 1)
                top3_correct_num += topk_correct_num(predict_y, test_label, 3)
                test_sample_num += test_label.size(0)
            # calculate the accuracy and print it
            top1_accuracy = top1_correct_num / test_sample_num
            top3_accuracy = top3_correct_num / test_sample_num
            display_accuracies_list.append(top1_accuracy)
            # discard the first 4 data
            local_top1_accuracy.append(top1_accuracy)
            local_top3_accuracy.append(top3_accuracy)
            print('top1 accuracy: {:.3f}'.format(top1_accuracy), flush=True)
            print('top3 accuracy: {:.3f}'.format(top3_accuracy), flush=True)
            # check and update architecture
            if np.abs(top1_accuracy - prev_accuracy) < 1e-4:
                if update_times < 0:
                    # save the module
                    if not os.path.isdir("models"):
                        os.mkdir("models")
                    torch.save(model, 'models/mnist_{:d}.pkl'.format(train_id))
                    final_accuracies_list.append(top1_accuracy)
                    print("Single training have %f top1 accuracy using model with architexcture [%d, %d, %d, %d, %s, %s, %s, %s, %s]" %(top1_accuracy, model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features, model.fc2.out_features, str(model.conv1_activation_func), str(model.conv2_activation_func), str(model.fc1_activation_func), str(model.fc2_activation_func), str(model.fc3_activation_func)))
                    break
                else:
                    # otherwise, update architecture
                    update_times -= 1
                    # find the potential best architecture\
                    model = copy.deepcopy(generate_architecture(model, local_top1_accuracy, local_top3_accuracy, generate_num, dev_num))
                    sgd = SGD(model.parameters(), lr = 1e-1)
                    local_top1_accuracy.clear()
                    local_top3_accuracy.clear()
                    # print model to help debug
                    print('%d, %d, %d, %d, %d' %(model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features, model.fc2.out_features, model.fc3.out_features))
    
            # if reach all epochs
            if current_epoch_num == max_train_epoches_num - 1:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(model, 'models/mnist_{:d}.pkl'.format(train_id))
                final_accuracies_list.append(top1_accuracy)
                print("Single training have %f top1 accuracy using model with architexcture [%d, %d, %d, %d, %s, %s, %s, %s, %s]" %(top1_accuracy, model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features, model.fc2.out_features, str(model.conv1_activation_func), str(model.conv2_activation_func), str(model.fc1_activation_func), str(model.fc2_activation_func), str(model.fc3_activation_func)))
            prev_accuracy = top1_accuracy
    
        # visualization
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(final_accuracies_list), torch.Tensor(train_list),
                            win=cur_batch_window, name='Training ID %d' %train_id,
                            update=(None if cur_batch_window is None else 'append'),
                            opts=cur_batch_window_opts)


def generate_architecture(model, local_top1_accuracy, local_top3_accuracy, generate_num, dev_num):
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
    FLOPs_list.append(LeNet.get_FLOPs(model))
    parameter_num_list.append(LeNet.get_parameter_num(model))

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
            loss_list, batch_list = [], []
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # clear the gradient data and update parameters based on error
                sgd.zero_grad()
                # get predict y and compute the error
                predict_y = dev_model(train_x.float())
                loss = loss_function(predict_y, train_label.long())
                # update visualization
                loss_list.append(loss.detach().cpu().item())
                batch_list.append(idx + 1)
                loss.backward()
                sgd.step()
            
            # initialize the testing parameters
            top1_correct_num = 0
            top3_correct_num = 0
            test_sample_num = 0

            # begin testing
            dev_model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
                predict_y = dev_model(test_x.float()).detach()
                top1_correct_num += topk_correct_num(predict_y, test_label, 1)
                top3_correct_num += topk_correct_num(predict_y, test_label, 3)
                test_sample_num += test_label.size(0)
            # calculate the accuracy and print it
            top1_accuracy = top1_correct_num / test_sample_num
            top3_accuracy = top3_correct_num / test_sample_num
            # discard the first half data as model need retraining
            if (dev_id + 1) % dev_num >= math.ceil(dev_num / 2):
                dev_top1_accuracies.append(top1_accuracy)
                dev_top3_accuracies.append(top3_accuracy)
        # store the model and score
        model_list.append(dev_model)
        top1_accuracy_list.append(dev_top1_accuracies)
        top3_accuracy_list.append(dev_top3_accuracies)
        FLOPs_list.append(LeNet.get_FLOPs(dev_model))
        parameter_num_list.append(LeNet.get_parameter_num(dev_model))
    score_list = compute_score(model_list, top1_accuracy_list, top3_accuracy_list, FLOPs_list, parameter_num_list)
    best_model_index = np.argmax(score_list)
    model = copy.deepcopy(model_list[best_model_index])
    print("model %d wins with %d conv1_kernel_num and %d conv2_kernel_num" %(best_model_index, model.conv1.out_channels, model.conv2.out_channels))
    return model


def compute_score(model_list, top1_accuracy_list, top3_accuracy_list, FLOPs_list, parameter_num_list):
    print(top1_accuracy_list)
    score_list = []
    # use softmax to process the FLOPs_list and parameter_num_list
    FLOPs_tensor = torch.tensor(FLOPs_list)
    parameter_num_tensor = torch.tensor(parameter_num_list)
    FLOPs_scaled = (FLOPs_tensor - torch.min(FLOPs_tensor)) / (torch.max(FLOPs_tensor) - torch.min(FLOPs_tensor))
    parameter_num_scaled = (parameter_num_tensor - torch.min(parameter_num_tensor)) / (torch.max(parameter_num_tensor) - torch.min(parameter_num_tensor))
    for model_id in range(len(model_list)):
        top1_accuracy = top1_accuracy_list[model_id]
        top3_accuracy = top3_accuracy_list[model_id]
        top1_accuracy = top1_accuracy[len(top1_accuracy) - 1]
        top3_accuracy = top3_accuracy[len(top3_accuracy) - 1]
        if top1_accuracy > 0.95:
            score_list.append(top1_accuracy * 0.5 +  0.5 - FLOPs_scaled[model_id] * 0.25 - parameter_num_scaled[model_id] * 0.25)
        else:
            score_list.append(top1_accuracy * 0.9 +  top3_accuracy * 0.1)
    print(FLOPs_scaled)
    print(score_list)
    return score_list
    

def topk_correct_num(predict_y, test_label, topk = 1):
    _, pred = predict_y.topk(topk, dim=1, largest=True, sorted=True)
    correct = pred.eq(test_label.view(-1, 1).expand_as(pred))
    correct_k = correct.any(dim=1).float().sum().item()
    return correct_k


if __name__ == '__main__':
    # define visualization parameters
    viz = visdom.Visdom(env=u'LeNet Module', use_incoming_socket=False)
    cur_batch_window = None
    cur_batch_window_opts = {
        'title': 'Accuracies during Training',
        'xlabel': 'Epoch Number',
        'ylabel': 'Epochs Accuracies',
        'width': 1200,
        'height': 600,
    }
    # train_paperversion()
    train_dynamic_architecture()
    print("Model finished training")