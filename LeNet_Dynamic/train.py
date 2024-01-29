from LeNet_Module import LeNet
import numpy as np
import os
import torch
import visdom
import copy
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# define visualization parameters
viz = visdom.Visdom(env=u'LeNet Module', use_incoming_socket=False)
cur_batch_window = None
cur_batch_window_opts = {
    'title': 'Converged Accuracies',
    'xlabel': 'Train Number',
    'ylabel': 'Converged Accuracy',
    'width': 1200,
    'height': 600,
}

def train_with_fixed_architecture():
    # load train and test data
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize the training parameters
    train_epoches_num = 1000
    global cur_batch_window
    train_num = 100
    train_list = []
    converged_accuracies = []

    for train_id in range(train_num):
        train_list.append(train_id)
        prev_accuracy = 0
        # reload the model
        model = LeNet().to(device)
        # define optimization method
        sgd = SGD(model.parameters(), lr = 1e-1)
        loss_function = CrossEntropyLoss()
        for current_epoch_num in range(train_epoches_num):
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
            test_correct_num = 0
            test_sample_num = 0

            # begin testing
            model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
                predict_y = model(test_x.float()).detach()
                predict_y = torch.argmax(predict_y, dim=-1)
                # compute the corret number and total number
                current_correct_num = predict_y == test_label
                test_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                test_sample_num += current_correct_num.shape[0]
            # calculate the accuracy and print it
            accuracy = test_correct_num / test_sample_num
            print('accuracy: {:.3f}'.format(accuracy), flush=True)
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model, 'models/mnist_{:.3f}.pkl'.format(accuracy))
            # if accuracy is small enough, finish
            if np.abs(accuracy - prev_accuracy) < 1e-4:
                converged_accuracies.append(accuracy)
                print("Single training finished with accuracy %f" %accuracy)
                break
            prev_accuracy = accuracy
    
    # visualization
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(converged_accuracies), torch.Tensor(train_list),
                            win=cur_batch_window, name='Train with fixed architecture',
                            update=(None if cur_batch_window is None else 'append'),
                            opts=cur_batch_window_opts)


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
    train_epoches_num = 1000
    global cur_batch_window
    train_num = 100
    train_list = []
    converged_accuracies = []

    for train_id in range(train_num):
        prev_accuracy = 0
        train_list.append(train_id)
        # reload the model each time
        model = LeNet().to(device)
        sgd = SGD(model.parameters(), lr = 1e-1)
        loss_function = CrossEntropyLoss()
        prev_score = -1.0
        local_accuracy = []
        for current_epoch_num in range(train_epoches_num):
            # check and update architecture
            if (current_epoch_num + 1) % 5 == 0 and max(local_accuracy) > 0.5:
                # compute the score
                cur_score = compute_score(local_accuracy, model.conv1.out_channels, model.conv2.out_channels)
                # clear the local accuracy
                local_accuracy.clear()
                if cur_score > prev_score:
                    # update score only if current architecture is better
                    prev_score = cur_score
                    old_model = copy.deepcopy(model)
                    LeNet.resize_kernel_num(model)
                else:
                    # restore the architecture
                    print('Reload the model')
                    model = old_model
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
            test_correct_num = 0
            test_sample_num = 0

            # begin testing
            model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                # move test data to device
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                # get predict y and predict its class
                predict_y = model(test_x.float()).detach()
                predict_y = torch.argmax(predict_y, dim=-1)
                # compute the corret number and total number
                current_correct_num = predict_y == test_label
                test_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                test_sample_num += current_correct_num.shape[0]
            # calculate the accuracy and print it
            accuracy = test_correct_num / test_sample_num
            local_accuracy.append(accuracy)
            print('accuracy: {:.3f}'.format(accuracy), flush=True)
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model, 'models/mnist_{:.3f}.pkl'.format(accuracy))
            # if accuracy is small enough, finish
            if np.abs(accuracy - prev_accuracy) < 1e-4:
                converged_accuracies.append(accuracy)
                print("Single training finished with accuracy %f" %accuracy)
                break
            prev_accuracy = accuracy
    
    # visualization
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(converged_accuracies), torch.Tensor(train_list),
                            win=cur_batch_window, name='Train with dynamic architecture',
                            update=(None if cur_batch_window is None else 'append'),
                            opts=cur_batch_window_opts)


def compute_score(local_accuracy, conv1_kernel_num, conv2_kernel_num):
    average_accuracies = sum(local_accuracy) / len(local_accuracy)
    min_accuracy = min(local_accuracy)
    if torch.rand(1).item() < 0.6:
        cur_model_score = average_accuracies * 1000 - conv1_kernel_num * conv2_kernel_num / 2
    else:
        cur_model_score = average_accuracies * 1000 - min_accuracy
    return cur_model_score


if __name__ == '__main__':
    train_with_fixed_architecture()
    train_dynamic_architecture()
    print("Model finished training")


