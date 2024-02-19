from LeNet_Module import LeNet
import numpy as np
import os
import torch
import visdom
import copy
import math
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def train_dynamic_architecture():
    # load train and test data
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize the training parameters
    max_train_epoches_num = 1000
    global cur_batch_window
    train_num = 3
    generate_num = 4
    dev_num = 8
    update_architecture_period = 10

    for train_id in range(train_num):
        prev_accuracy = 0
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
            # check and update architecture
            if current_epoch_num != 0 and (current_epoch_num) % update_architecture_period == 0 and max(local_top1_accuracy) > 0.5:
                # find the potential best architecture\
                model = copy.deepcopy(generate_architecture(model, local_top1_accuracy, local_top3_accuracy, generate_num, dev_num))
                sgd = SGD(model.parameters(), lr = 1e-1)
                local_top1_accuracy.clear()
                local_top3_accuracy.clear()
                # print model to help debug
                print('%d, %d, %d, %d, %d' %(model.conv1.out_channels, model.conv2.out_channels, model.fc1.out_features, model.fc2.out_features, model.fc3.out_features))
            
            # begin training
            model.train()               # set model into training
            loss_list, batch_list = [], []
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # get predict y and compute the error
                predict_y = model(train_x.float())
                loss = loss_function(predict_y, train_label.long())
                # update visualization
                loss_list.append(loss.detach().cpu().item())
                batch_list.append(idx + 1)
                # clear the gradient data and update parameters based on error
                sgd.zero_grad()
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
            if (current_epoch_num + 1) % update_architecture_period >= math.ceil(update_architecture_period / 2):
                local_top1_accuracy.append(top1_accuracy)
                local_top3_accuracy.append(top3_accuracy)
            print('top1 accuracy: {:.3f}'.format(top1_accuracy), flush=True)
            print('top3 accuracy: {:.3f}'.format(top3_accuracy), flush=True)
            # if accuracy is small enough, finish
            if np.abs(top1_accuracy - prev_accuracy) < 1e-4 and current_epoch_num >= 20:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(model, 'models/mnist_{:.3f}.pkl'.format(top1_accuracy))
                print("Single training have %f top1 accuracy using model with %d Flops and %d parameternum" %(top1_accuracy, LeNet.get_FLOPs(model), LeNet.get_parameter_num(model)))
                break
            prev_accuracy = top1_accuracy
    
        # visualization
        if viz.check_connection():
            cur_batch_window = viz.line(torch.Tensor(display_accuracies_list), torch.Tensor(epoch_list),
                                win=cur_batch_window, name='Training ID %d' %train_id,
                                update=(None if cur_batch_window is None else 'append'),
                                opts=cur_batch_window_opts)


def generate_architecture(model, local_top1_accuracy, local_top3_accuracy, generate_num, dev_num):
    # load train and test data
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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
        # generate architecture
        dev_model = copy.deepcopy(original_model)
        LeNet.update_architecture(dev_model)
        dev_model = dev_model.to(device)
        sgd = SGD(dev_model.parameters(), lr = 1e-1)
        dev_top1_accuracies = []
        dev_top3_accuracies = []
        # train the architecture for dev_num times
        for dev_id in range(dev_num):
            # begin training
            dev_model.train()               # set model into training
            loss_list, batch_list = [], []
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # get predict y and compute the error
                predict_y = dev_model(train_x.float())
                loss = loss_function(predict_y, train_label.long())
                # update visualization
                loss_list.append(loss.detach().cpu().item())
                batch_list.append(idx + 1)
                # clear the gradient data and update parameters based on error
                sgd.zero_grad()
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
    score_list = []
    # use softmax to process the FLOPs_list and parameter_num_list
    FLOPs_tensor = torch.tensor(FLOPs_list)
    parameter_num_tensor = torch.tensor(parameter_num_list)
    FLOPs_softmax = torch.square(FLOPs_tensor) / torch.sum(torch.square(FLOPs_tensor), dim = 0, keepdim = True)
    parameter_num_softmax = torch.square(parameter_num_tensor) / torch.sum(torch.square(parameter_num_tensor), dim = 0, keepdim = True)
    print(FLOPs_softmax)
    for model_id in range(len(model_list)):
        top1_accuracy = top1_accuracy_list[model_id]
        top3_accuracy = top3_accuracy_list[model_id]
        top1_accuracy = sum(top1_accuracy) / len(top1_accuracy)
        top3_accuracy = sum(top3_accuracy) / len(top3_accuracy)
        if torch.rand(1).item() < 0.6:
            score_list.append(top1_accuracy * 0.8 +  top3_accuracy * 0.1 - FLOPs_softmax[model_id] * 0.05 - parameter_num_softmax[model_id] * 0.05)
        else:
            score_list.append(top1_accuracy * 0.8 +  top3_accuracy * 0.2)
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
    train_dynamic_architecture()
    print("Model finished training")


