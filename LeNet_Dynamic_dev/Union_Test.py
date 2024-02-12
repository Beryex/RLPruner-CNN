from LeNet_Module import LeNet
import LeNet_Module
import numpy as np
import os
import torch
import visdom
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# define visualization parameters
viz = visdom.Visdom(env=u'LeNet Module', use_incoming_socket=False)
cur_batch_window = None
cur_batch_window_opts = {
    'title': 'Train Final Accuracy',
    'xlabel': 'Train Number',
    'ylabel': 'Final Converged Accuracy',
    'width': 1200,
    'height': 600,
}

def train_weights_bias(accuracies, total_train_num):

    global cur_batch_window
    cur_batch_window = None
    train_num = 1
    cur_accuracies_list = []
    train_list = []

    for cur_train_num in range(train_num):
        train_list.append(cur_train_num + 1)
        # load train and test data
        batch_size = 256
        train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
        test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        # move the LeNet Module into the corresponding device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LeNet(conv1_kernel_num_i= LeNet_Module.conv1_kernel_num, conv2_kernel_num_i= LeNet_Module.conv2_kernel_num).to(device)
        # define optimization method
        sgd = SGD(model.parameters(), lr = 1e-1)
        loss_function = CrossEntropyLoss()
        # initialize the training parameters
        train_epoches_num = 100
        prev_accuracy = 0
        for current_epoch_num in range(train_epoches_num):
            # begin training
            model.train()               # set model into training
            for idx, (train_x, train_label) in enumerate(train_loader):
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # get predict y and compute the error
                predict_y = model(train_x.float())
                loss = loss_function(predict_y, train_label.long())
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
            cur_accuracy = test_correct_num / test_sample_num
            print('accuracy: {:.3f}'.format(cur_accuracy), flush=True)
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model, 'models/mnist_{:.3f}.pkl'.format(cur_accuracy))
            # if accuracy is small enough, upload average accuracy
            if np.abs(cur_accuracy - prev_accuracy) < 1e-4:
                cur_accuracies_list.append(cur_accuracy)
                print("Single training finished with accuracy %f" %cur_accuracy)
                break
            prev_accuracy = cur_accuracy
    
    # compute final average accuracies
    average_accuracies = sum(cur_accuracies_list) / len(cur_accuracies_list)
    accuracies.append(average_accuracies)
    print("The final average accuracies is %f" %average_accuracies)
    # visualization this training final accuracy
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(cur_accuracies_list), torch.Tensor(train_list),
                            win=cur_batch_window, name='train' + str(total_train_num),
                            update=(None if cur_batch_window is None else 'replace'),
                            opts=cur_batch_window_opts)

if __name__ == '__main__':
    accuracies = []
    train_weights_bias(accuracies, 0)
    LeNet_Module.conv1_kernel_num = 7
    LeNet_Module.conv2_kernel_num = 20
    train_weights_bias(accuracies, 1)
    print("Model finished training")




