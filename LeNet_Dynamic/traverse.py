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


def traverse_architecture():
    # load train and test data
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize the training parameters
    train_epoches_num = 1000
    global cur_batch_window
    train_num = 5
    architecture_list = []
    scores_list = []

    for propotion in np.arange(0.1, 1, 0.1):
        for architecture_id in range(1, 30, 1):
            architecture_list.append(architecture_id)
            converged_accuracies = []
            for train_id in range(train_num):
                # reload the model each time
                model = LeNet(conv1_kernel_num_i= math.ceil(architecture_id * propotion), conv2_kernel_num_i= architecture_id).to(device)
                sgd = SGD(model.parameters(), lr = 1e-1)
                loss_function = CrossEntropyLoss()
                prev_accuracy = -1
                for current_epoch_num in range(train_epoches_num):
                    # check and update architecture
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
                    # print('accuracy: {:.3f}'.format(accuracy), flush=True)
                    # if accuracy is small enough, finish
                    if np.abs(accuracy - prev_accuracy) < 1e-4:
                        # save the module
                        if not os.path.isdir("models"):
                            os.mkdir("models")
                        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(accuracy))
                        converged_accuracies.append(accuracy)
                        print("Single training finished with architecture [%d, %d, %f] and accuracy %f" %(math.ceil(architecture_id * propotion), architecture_id, propotion, accuracy))
                        break
                    prev_accuracy = accuracy
            scores_list.append(compute_score(converged_accuracies, architecture_id, propotion))
        
        # visualization
        if viz.check_connection():
            cur_batch_window = viz.line(torch.Tensor(scores_list), torch.Tensor(architecture_list),
                                win=cur_batch_window, name='Score for architecture with propotion %f' %propotion,
                                update=(None if cur_batch_window is None else 'append'),
                                opts=cur_batch_window_opts)


def compute_score(converged_accuracies, architecture_id, propotion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet(conv1_kernel_num_i= math.ceil(architecture_id * propotion), conv2_kernel_num_i= architecture_id).to(device)
    average_accuracy = sum(converged_accuracies) / len(converged_accuracies)
    accuracy_std_dev = np.std(converged_accuracies)
    FLOPs = LeNet.get_FLOPs(model)
    parameter_num = LeNet.get_parameter_num(model)
    print("model %d have %d Flops and %d parameternum with %f average accuracies and %f standard deviation" %(architecture_id, FLOPs, parameter_num, average_accuracy, accuracy_std_dev))
    return average_accuracy * 1000 - accuracy_std_dev - FLOPs / 10000 - parameter_num / 500



if __name__ == '__main__':
    # define visualization parameters
    viz = visdom.Visdom(env=u'LeNet Module', use_incoming_socket=False)
    cur_batch_window = None
    cur_batch_window_opts = {
        'title': 'Score for different architecture',
        'xlabel': 'Train Number',
        'ylabel': 'Score',
        'width': 1200,
        'height': 600,
    }
    traverse_architecture()
    print("Model finished training")


