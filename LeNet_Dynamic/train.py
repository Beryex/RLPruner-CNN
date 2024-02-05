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
    train_epoches_num = 1000
    global cur_batch_window
    train_num = 3
    possible_model_num = 4
    dev_num = 8

    for train_id in range(train_num):
        prev_accuracy = 0
        # reload the model each time
        model = LeNet().to(device)
        sgd = SGD(model.parameters(), lr = 1e-1)
        loss_function = CrossEntropyLoss()
        prev_score = -1.0
        local_accuracy = []
        epoch_list = []
        accuracies_list = []
        for current_epoch_num in range(train_epoches_num):
            epoch_list.append(current_epoch_num)
            # check and update architecture
            if current_epoch_num != 0 and (current_epoch_num) % 10 == 0 and max(local_accuracy) > 0.5:
                # find the potential best architecture
                model_list = []
                score_list = []
                dev_accuracies = []
                cur_score = compute_score(local_accuracy, model.conv1.out_channels, model.conv2.out_channels)
                model_list.append(model)
                score_list.append(cur_score)
                original_model = copy.deepcopy(model)
                # clear the local accuracy
                local_accuracy.clear()
                for model_id in range(possible_model_num):
                    dev_model = copy.deepcopy(original_model)
                    LeNet.update_architecture(dev_model)
                    dev_model = dev_model.to(device)
                    sgd = SGD(dev_model.parameters(), lr = 1e-1)
                    for dev_id in range(dev_num):
                        # begin training
                        dev_model.train()               # set model into training
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
                        dev_model.eval()
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
                        # discard the first 4 data
                        if (dev_id + 1) % 8 >= 4:
                            dev_accuracies.append(accuracy)
                    # store the model and score
                    model_list.append(dev_model)
                    cur_score = compute_score(dev_accuracies, dev_model.conv1.out_channels, dev_model.conv2.out_channels)
                    score_list.append(cur_score)
                best_model_index = np.argmax(score_list)
                model = copy.deepcopy(model_list[best_model_index])
                print("model %d wins with %d conv1_kernel_num and %d conv2_kernel_num" %(best_model_index, model.conv1.out_channels, model.conv2.out_channels))
                sgd = SGD(model.parameters(), lr = 1e-1)
            
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
            accuracies_list.append(accuracy)
            # discard the first 4 data
            if (current_epoch_num + 1) % 10 >= 5:
                local_accuracy.append(accuracy)
            print('accuracy: {:.3f}'.format(accuracy), flush=True)
            # if accuracy is small enough, finish
            if np.abs(accuracy - prev_accuracy) < 1e-4 and current_epoch_num >= 20:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(model, 'models/mnist_{:.3f}.pkl'.format(accuracy))
                print("Single training have %f average accuracy using model with %d Flops and %d parameternum" %(accuracy, LeNet.get_FLOPs(model), LeNet.get_parameter_num(model)))
                break
            prev_accuracy = accuracy
    
        # visualization
        if viz.check_connection():
            cur_batch_window = viz.line(torch.Tensor(accuracies_list), torch.Tensor(epoch_list),
                                win=cur_batch_window, name='Training ID %d' %train_id,
                                update=(None if cur_batch_window is None else 'append'),
                                opts=cur_batch_window_opts)


def compute_score(local_accuracy, conv1_kernel_num, conv2_kernel_num):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet(conv1_kernel_num_i= conv1_kernel_num, conv2_kernel_num_i= conv2_kernel_num).to(device)
    average_accuracy = sum(local_accuracy) / len(local_accuracy)
    accuracy_std_dev = np.std(local_accuracy)
    FLOPs = LeNet.get_FLOPs(model)
    parameter_num = LeNet.get_parameter_num(model)
    if torch.rand(1).item() < 0.6:
        return average_accuracy * 1000 - accuracy_std_dev - FLOPs / 10000 - parameter_num / 500
    else:
        return average_accuracy * 1000 + max(local_accuracy) * 50 - FLOPs / 10000 - parameter_num / 500


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


