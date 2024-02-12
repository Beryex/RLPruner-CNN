from model import Model
import numpy as np
import os
import torch
import visdom
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
    train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
    all_epoch = 1000
    train_time = 2
    final_accuracies_list = []
    train_list = []
    for train_id in range(train_time):
        train_list.append(train_id)
        model = Model().to(device)
        sgd = SGD(model.parameters(), lr=5e-2)
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
            if np.abs(acc - prev_acc) < 1e-4 and current_epoch >= 40:
                final_accuracies_list.append(acc)
                break
            prev_acc = acc
    # visualization
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(final_accuracies_list), torch.Tensor(train_list),
                            win=cur_batch_window, name='Paper Version',
                            update=(None if cur_batch_window is None else 'append'),
                            opts=cur_batch_window_opts)


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
    train_paperversion()
    print("Model finished training")