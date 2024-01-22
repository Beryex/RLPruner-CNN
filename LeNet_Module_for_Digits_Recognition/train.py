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

# define model parameters max value
max_conv1_kernel_num = 20
max_conv2_kernel_num = 60
max_training_times = 30

def train_weights_bias(model_architecture, model_training_result, training_id):

    global cur_batch_window
    training_num = 10
    cur_accuracies_list = []
    train_list = []
    epoches_count = 0

    for cur_train_num in range(training_num):
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
                epoches_count += current_epoch_num + 1
                break
            prev_accuracy = cur_accuracy
    
    # visualization this training final accuracy
    if viz.check_connection():
        cur_batch_window = viz.line(torch.Tensor(cur_accuracies_list), torch.Tensor(train_list),
                            win=cur_batch_window, name='[' + str(model_architecture[0]) + ',' + str(model_architecture[1]) + ',' + str(training_id) + ']',
                            update=(None if cur_batch_window is None else 'append'),
                            opts=cur_batch_window_opts)
    # compute final average accuracies and average epoches needed
    final_average_accuracies = sum(cur_accuracies_list) / len(cur_accuracies_list)
    epoches_average_needed = epoches_count / 10
    model_training_result[model_architecture[0]][model_architecture[1]]['final average accuracies'] = final_average_accuracies
    model_training_result[model_architecture[0]][model_architecture[1]]['epoches average needed'] = epoches_average_needed
    cur_model_score = compute_score(final_average_accuracies, epoches_average_needed)
    model_training_result[model_architecture[0]][model_architecture[1]]['score'] = cur_model_score
    print("The final average accuracies is %f" %final_average_accuracies)
    print("The epoches average needed is %f" %epoches_average_needed)
    print("The model has score %f" %cur_model_score)
    return cur_model_score


def compute_score(final_average_accuracies, epoches_average_needed):
    cur_model_score = final_average_accuracies * 1000 - epoches_average_needed * 5
    return cur_model_score


# return 0 when update successfully, return -1 when error occurs
def update_model_architecture(model_training_result):
    cur_conv1_kernel_num = LeNet_Module.conv1_kernel_num
    cur_conv2_kernel_num = LeNet_Module.conv2_kernel_num
    # update conv2 first, if its size doesn't match conv1, then update conv1
    if cur_conv2_kernel_num <= 1 or cur_conv2_kernel_num >= max_conv2_kernel_num or cur_conv1_kernel_num <= 1 or cur_conv1_kernel_num >= max_conv1_kernel_num:
        print('Need to increase the size of model architecture!')
        return -1
    # prefer to move larger architecture
    if model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num - 1]['score'] == 0 and model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num + 1]['score'] == 0:
        LeNet_Module.conv2_kernel_num += 1
        if(LeNet_Module.conv2_kernel_num > LeNet_Module.conv1_kernel_num * 3.8):
            LeNet_Module.conv1_kernel_num +=1
        return 0
    if model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num + 1]['score'] > model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num]['score']:
        LeNet_Module.conv2_kernel_num += 1
        if(LeNet_Module.conv2_kernel_num > LeNet_Module.conv1_kernel_num * 3.8):
            LeNet_Module.conv1_kernel_num +=1
        return 0
    if model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num - 1]['score'] > model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num]['score']:
        LeNet_Module.conv2_kernel_num -= 1
        if(LeNet_Module.conv2_kernel_num < LeNet_Module.conv1_kernel_num * 2.2):
            LeNet_Module.conv1_kernel_num -=1
        return 0
    if model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num - 1]['score'] < model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num]['score'] and model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num + 1]['score'] == 0:
        LeNet_Module.conv2_kernel_num += 1
        if(LeNet_Module.conv2_kernel_num > LeNet_Module.conv1_kernel_num * 3.8):
            LeNet_Module.conv1_kernel_num +=1
        return 0
    if model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num - 1]['score'] == 0 and model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num + 1]['score'] < model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num]['score']:
        LeNet_Module.conv2_kernel_num -= 1
        if(LeNet_Module.conv2_kernel_num < LeNet_Module.conv1_kernel_num * 2.2):
            LeNet_Module.conv1_kernel_num -= 1
        return 0
    print('No updates occur. This is exception that should never happen')
    return -1


def check_architecture(model_training_result):
    cur_conv1_kernel_num = LeNet_Module.conv1_kernel_num
    cur_conv2_kernel_num = LeNet_Module.conv2_kernel_num
    if model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num - 1]['score'] < model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num]['score'] and model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num + 1]['score'] < model_training_result[cur_conv1_kernel_num][cur_conv2_kernel_num]['score']:
        print('Local optimal found when updating architecture')
        return 1
if __name__ == '__main__':
    # define how to evaluate the model
    dtype = [('final average accuracies', float), ('epoches average needed', float), ('score', float)]
    model_training_result = np.zeros((max_conv1_kernel_num + 1, max_conv2_kernel_num + 1), dtype=dtype)
    training_id = 0
    prev_model_score = 0
    for training_id in range(max_training_times):
        model_architecture = np.array([LeNet_Module.conv1_kernel_num, LeNet_Module.conv2_kernel_num])
        cur_model_score = train_weights_bias(model_architecture, model_training_result, training_id)
        # if score converge, stop training
        if np.abs(cur_model_score - prev_model_score) < 1:
            print('The local optimal architure has %f conv1_kernel_num, %f conv2_kernel_num' %(model_architecture[0], model_architecture[1]))
            break
        update_result = update_model_architecture(model_training_result)
        if update_result == -1:
            print('Update model architecture error occurs')
            break
        if check_architecture(model_training_result) == 1:
            print('The local optimal architure has %f conv1_kernel_num, %f conv2_kernel_num' %(model_architecture[0], model_architecture[1]))
            break
        prev_model_score = cur_model_score
    print("Model finished training")




