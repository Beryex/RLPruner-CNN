from LeNet_Module import LeNet
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    # load train and test data
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # move the LeNet Module into the corresponding device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet().to(device)

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
        accuracy = test_correct_num / test_sample_num
        print('accuracy: {:.3f}'.format(accuracy), flush=True)
        # save the module
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(accuracy))
        # if accuracy is small enough, finish
        if np.abs(accuracy - prev_accuracy) < 1e-4:
            break
        prev_accuracy = accuracy
    print("Model finished training")




