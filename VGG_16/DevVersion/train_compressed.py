import os
import time
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import copy
from thop import profile

from conf import settings
from models.vgg import VGG, Custom_Conv2d, Custom_Linear
from utils import (get_CIFAR10_training_dataloader, get_CIFAR10_test_dataloader, 
                   get_CIFAR10_dev_training_dataloader, get_CIFAR100_training_dataloader, 
                   get_CIFAR100_test_dataloader, get_CIFAR100_dev_training_dataloader, WarmUpLR,
                   count_custom_conv2d, count_custom_linear)

def prune_architecture(top1_acc, top5_acc):
    global net
    global tolerance_times
    global modification_num
    global generate_num
    global optimizer
    global accuracy_threshold
    global strategy
    if tolerance_times >= 0:
        net, model_index = copy.deepcopy(generate_architecture(net, top1_acc, top5_acc))
        net = net.to(device)
        if model_index == 0:
            tolerance_times -= 1
            if eap == True:
                if tolerance_times in settings.TOLERANCE_MILESTONES:
                    modification_num //= 2
                    generate_num += 1
            else:
                modification_num //= 2
                generate_num += 1
        optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
        # save the module
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, 'models/VGG_Compressed_Pruned_{:d}.pkl'.format(current_time))
    else:
        if args.criteria == 'compression' and Para_compression_ratio >= compression_threshold:
            # decrement accuracy_threshold and reinitialize hyperparameter if criteria is compression ratio
            accuracy_threshold -= 0.05
            generate_num = settings.MAX_GENERATE_NUM
            modification_num = settings.MAX_PRUNE_NUM
            tolerance_times = settings.MAX_TOLERANCE_TIMES
            logging.info('Current accuracy threshold: {}'.format(accuracy_threshold))
        else:
            strategy = 'finished'
            generate_num = settings.MAX_GENERATE_NUM
            modification_num = settings.MAX_QUANTIZE_NUM
            tolerance_times = settings.MAX_TOLERANCE_TIMES
            # save the module and break
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, 'models/VGG_Compressed_Pruned_{:d}.pkl'.format(current_time))

def quantize_architecture(top1_acc, top5_acc):
    global net
    global tolerance_times
    global modification_num
    global generate_num
    global optimizer
    global accuracy_threshold
    global strategy
    if tolerance_times >= 0:
        net, model_index = copy.deepcopy(generate_architecture(net, top1_acc, top5_acc))
        net = net.to(device)
        if model_index == 0:
            tolerance_times -= 1
            if eap == True:
                if tolerance_times in settings.TOLERANCE_MILESTONES:
                    modification_num //= 2
                    generate_num += 1
            else:
                modification_num //= 2
                generate_num += 1
        optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
        # save the module
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, 'models/VGG_Compressed_Quantized_{:d}.pkl'.format(current_time))
    else:
        strategy = 'finished'
        # save the module and break
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(net, 'models/VGG_Compressed_Quantized_{:d}.pkl'.format(current_time))


def train(epoch):
    net.train()
    with tqdm(total=len(cifar10_training_loader.dataset), desc=f'Epoch {epoch}/{settings.DYNAMIC_EPOCH}', unit='img') as pbar:
        for batch_index, (images, labels) in enumerate(cifar10_training_loader):
            labels = labels.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.update(images.shape[0])
            pbar.set_postfix(**{'loss (batch)': loss.item()})

@torch.inference_mode()
def eval_training(epoch=0):
    net.eval()
    test_loss = 0.0
    correct_1 = 0.0
    correct_5 = 0.0
    for (images, labels) in tqdm(cifar10_test_loader, total=len(cifar10_test_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.cuda()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    top1_acc = correct_1 / len(cifar10_test_loader.dataset)
    top5_acc = correct_5 / len(cifar10_test_loader.dataset)
    logging.info('Top1 Accuracy: {}, Top5 Accuracy: {}'.format(top1_acc, top5_acc))
    return top1_acc, top5_acc


def generate_architecture(model, local_top1_accuracy, local_top5_accuracy):
    # initialize all evaluating variables
    top1_accuracy_tensors = torch.zeros(2)
    top5_accuracy_tensors = torch.zeros(2)
    FLOPs_tensors = torch.zeros(2)
    parameter_num_tensors = torch.zeros(2)
    top1_accuracy_tensors[0] = local_top1_accuracy
    top5_accuracy_tensors[0] = local_top5_accuracy
    local_FLOPs, local_parameter_num = profile(model, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[0] = local_FLOPs
    parameter_num_tensors[0] = local_parameter_num

    # generate architecture
    dev_model, top1_pretrain_accuracy_tensors, prune_probability_distribution_tensors = copy.deepcopy(get_best_generated_architecture(model))
    dev_model = dev_model.to(device)
    dev_lr = lr
    dev_optimizer = optim.SGD(dev_model.parameters(), lr=dev_lr, momentum=0.9, weight_decay=5e-4)
    dev_warmup_scheduler = WarmUpLR(dev_optimizer, iter_per_epoch * warm)
    # train the architecture for dev_num times
    for dev_id in range(1, dev_num + 1):
        if dev_id in settings.DYNAMIC_MILESTONES:
            dev_lr *= gamma
            for param_group in dev_optimizer.param_groups:
                param_group['lr'] = dev_lr
        # begin training
        dev_model.train()               # set model into training
        with tqdm(total=len(cifar10_training_loader.dataset), desc=f'Training best generated Architecture Epoch {dev_id}/{dev_num}', unit='img', leave=False) as pbar:
            for train_x, train_label in cifar10_training_loader:
                # move train data to device
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                # clear the gradient data and update parameters based on error
                dev_optimizer.zero_grad()
                # get predict y and compute the error
                predict_y = dev_model(train_x)
                loss = loss_function(predict_y, train_label)
                # update visualization
                loss.backward()
                dev_optimizer.step()

                if dev_id <= warm:
                    dev_warmup_scheduler.step()
                
                pbar.update(train_label.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

    # initialize the testing parameters
    correct_1 = 0.0
    correct_5 = 0.0
    # begin testing
    dev_model.eval()
    with torch.inference_mode():
        for test_x, test_label in cifar10_test_loader:
            # move test data to device
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            # get predict y and predict its class
            outputs = dev_model(test_x)
            _, preds = outputs.topk(5, 1, largest=True, sorted=True)
            #compute top1
            correct_1 += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
            #compute top 5
            top5_correct = test_label.view(-1, 1).expand_as(preds) == preds
            correct_5 += top5_correct.any(dim=1).sum().item()
    # calculate the accuracy
    top1_accuracy_tensors[1] = correct_1 / len(cifar10_test_loader.dataset)
    top5_accuracy_tensors[1] = correct_5 / len(cifar10_test_loader.dataset)
    dev_FLOPs, dev_parameter_num = profile(dev_model, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    FLOPs_tensors[1] = dev_FLOPs
    parameter_num_tensors[1] = dev_parameter_num
    global Para_compression_ratio
    score_tensors = compute_score(top1_accuracy_tensors, top5_accuracy_tensors, FLOPs_tensors, parameter_num_tensors)
    best_model_index = torch.argmax(score_tensors)
    best_model_FLOPs = FLOPs_tensors[best_model_index]
    best_model_Params = parameter_num_tensors[best_model_index]
    FLOPs_compression_ratio = 1 - best_model_FLOPs / original_FLOPs_num
    Para_compression_ratio = 1 - best_model_Params / original_para_num

    if best_model_index == 0:
        logging.info('Original Model wins')
        if eap == True:
            model.update_prune_probability_distribution(top1_pretrain_accuracy_tensors, prune_probability_distribution_tensors, step_length, forward=False)
    else:
        # means generated is better
        model = dev_model
        logging.info('Generated Model wins')
        if eap == True:
            model.update_prune_probability_distribution(top1_pretrain_accuracy_tensors, prune_probability_distribution_tensors, step_length, forward=True)
    logging.info('Current prune probability distribution: {}'.format(model.prune_probability))
    logging.info('Current compression ratio: FLOPs: {}, Parameter number {}'.format(FLOPs_compression_ratio, Para_compression_ratio))
    
    return model, best_model_index


def get_best_generated_architecture(model):
    # initialize all evaluating variables
    model_list = []
    top1_pretrain_accuracy_tensors = torch.zeros(generate_num)
    prune_probability_distribution_tensors = torch.zeros(generate_num, model.prune_choices_num)

    original_model = copy.deepcopy(model)
    with tqdm(total=generate_num, desc=f'Generated architectures', unit='model', leave=False) as pbar:
        for model_id in range(generate_num):
            # generate architecture
            dev_model = copy.deepcopy(original_model)
            prune_probability_distribution_tensors[model_id] += VGG.update_architecture(dev_model, modification_num, strategy)
            dev_model = dev_model.to(device)
            dev_lr = lr
            dev_optimizer = optim.SGD(dev_model.parameters(), lr=dev_lr, momentum=0.9, weight_decay=5e-4)
            dev_warmup_scheduler = WarmUpLR(dev_optimizer, iter_per_epoch * warm)
            # train the architecture for dev_num times
            for dev_id in range(1, dev_pretrain_num + 1):
                if dev_id in settings.DYNAMIC_PRETRAIN_MILESTONES:
                    dev_lr *= gamma
                    for param_group in dev_optimizer.param_groups:
                        param_group['lr'] = dev_lr
                # begin training
                dev_model.train()               # set model into training
                for train_x, train_label in cifar10_dev_training_loader:
                    # move train data to device
                    train_x = train_x.to(device)
                    train_label = train_label.to(device)
                    # clear the gradient data and update parameters based on error
                    dev_optimizer.zero_grad()
                    # get predict y and compute the error
                    predict_y = dev_model(train_x)
                    loss = loss_function(predict_y, train_label)
                    # update visualization
                    loss.backward()
                    dev_optimizer.step()

                    if dev_id <= warm:
                        dev_warmup_scheduler.step()
                
            # initialize the testing parameters
            correct_1 = 0.0
            # begin testing
            dev_model.eval()
            with torch.inference_mode():
                for test_x, test_label in cifar10_test_loader:
                    # move test data to device
                    test_x = test_x.to(device)
                    test_label = test_label.to(device)
                    # get predict y and predict its class
                    outputs = dev_model(test_x)
                    _, preds = outputs.topk(5, 1, largest=True, sorted=True)
                    #compute top1
                    correct_1 += (preds[:, :1] == test_label.unsqueeze(1)).sum().item()
                # calculate the accuracy and print it
                top1_pretrain_accuracy_tensors[model_id] = correct_1 / len(cifar10_test_loader.dataset)
            pbar.update(1)
            # store the model and score
            model_list.append(dev_model)
    best_model_index = torch.argmax(top1_pretrain_accuracy_tensors)
    best_generated_model = model_list[best_model_index]

    logging.info('Pretrained Generated Model Top1 Accuracy List: {}'.format(top1_pretrain_accuracy_tensors))
    logging.info('Model {} wins'.format(best_model_index))
    return best_generated_model, top1_pretrain_accuracy_tensors, prune_probability_distribution_tensors


def compute_score(top1_accuracy_tensors, top5_accuracy_tensors, FLOPs_tensors, parameter_num_tensors):
    score_tensors = torch.zeros(2)
    # use Min-Max Normalization to process the FLOPs_list and parameter_num_list
    FLOPs_scaled_tensors = (FLOPs_tensors - torch.min(FLOPs_tensors)) / (torch.max(FLOPs_tensors) - torch.min(FLOPs_tensors))
    parameter_num_scaled_tensors = (parameter_num_tensors - torch.min(parameter_num_tensors)) / (torch.max(parameter_num_tensors) - torch.min(parameter_num_tensors))
    
    if torch.min(top1_accuracy_tensors) > accuracy_threshold:
        score_tensors = 1 - FLOPs_scaled_tensors * 0.5 - parameter_num_scaled_tensors * 0.5
    elif torch.max(top1_accuracy_tensors) < accuracy_threshold:
        score_tensors = top1_accuracy_tensors
    else:
        if top1_accuracy_tensors[1] > accuracy_threshold - 0.005:
            score_tensors[1] = 1.0
        else:
            score_tensors[0] = 1.0
    
    logging.info('Generated Model Top1 Accuracy List: {}'.format(top1_accuracy_tensors))
    logging.info('Generated Model Parameter Number Scaled List: {}'.format(parameter_num_tensors))
    logging.info('Generated Model Score List: {}'.format(score_tensors))
    return score_tensors

def check_args(args):
    if args.criteria == 'accuracy':
        if args.compression_threshold is not None:
            logging.error("--compression_threshold is not allowed when criteria is 'accuracy'")
            sys.exit(1)
        if args.accuracy_threshold is None:
            logging.error("--compression_threshold is not allowed when criteria is 'accuracy'")
            sys.exit(1)
    elif args.criteria == 'compression':
        if args.accuracy_threshold is not None:
            logging.error("--accuracy_threshold is not allowed when criteria is 'compression'")
            sys.exit(1)
        if args.compression_threshold is None:
            logging.error("--compression_threshold is not allowed when criteria is 'accuracy'")
            sys.exit(1)
    else:
        logging.error("--criteria must be 'accuracy' or 'compression'")
        sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description='Dynamic Compressing VGG16')
    parser.add_argument('--criteria', '-c', type=str, default='accuracy', help='Compressed the model with accuracy_threshold or compression_threshold')
    parser.add_argument('--accuracy_threshold', '-A', metavar='A', type=float, default=None, help='The final accuracy the architecture will achieve')
    parser.add_argument('--compression_threshold', '-C', metavar='C', type=float, default=None, help='The final compression ratio the architecture will achieve')
    parser.add_argument('--enable_adaptive_pruning', '-eap', action='store_true', default=False, help='Enable the special feature if set')

    args = parser.parse_args()
    check_args(args)

    return args


if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    logging.info('Start with random seed: {}'.format(current_time))

    # move the LeNet Module into the corresponding device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = settings.INITIAL_LR
    gamma = settings.LR_DECAY
    current_lr = lr * gamma * gamma * gamma
    step_length = settings.STEP_LENGTH
    warm = 1
    batch_size = settings.BATCH_SIZE
    generate_num = settings.MAX_GENERATE_NUM
    modification_num = settings.MAX_PRUNE_NUM
    eap = args.enable_adaptive_pruning
    if eap == True:
        tolerance_times = settings.MAX_TOLERANCE_TIMES_EAP
    else:
        tolerance_times = settings.MAX_TOLERANCE_TIMES
    strategy = "prune"
    dev_num = settings.DEV_NUM
    dev_pretrain_num = settings.DEV_PRETRAIN_NUM
    if args.criteria == 'accuracy':
        accuracy_threshold = args.accuracy_threshold
        compression_threshold = settings.DEFAULT_COMPRESSION_THRESHOLD
    else:
        accuracy_threshold = settings.DEFAULT_ACCURACY_THRESHOLD
        compression_threshold = args.compression_threshold

    #data preprocessing:
    cifar10_training_loader = get_CIFAR100_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    cifar10_test_loader = get_CIFAR100_test_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    cifar10_dev_training_loader = get_CIFAR100_dev_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    net = torch.load('models/VGG_Original_1713455040.pkl')  # replace it with the model gained by train_original.py
    net = net.to(device)
    net.prune_probability = torch.tensor([0.002, 0.002, 0.002, 0.002, 0.002, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.25, 0.25])

    input = torch.rand(1, 3, 32, 32).to(device)
    custom_ops = {Custom_Conv2d: count_custom_conv2d, Custom_Linear: count_custom_linear}
    original_FLOPs_num, original_para_num = profile(net, inputs = (input, ), verbose=False, custom_ops=custom_ops)
    Para_compression_ratio = 1.000

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(cifar10_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    
    for epoch in range(1, settings.DYNAMIC_EPOCH + 1):
        train(epoch)
        top1_acc, top5_acc = eval_training(epoch)

        # dynamic generate architecture
        if epoch % 5 == 0:
            if strategy == "prune":
                prune_architecture(top1_acc, top5_acc)
            elif strategy == "quantize":
                quantize_architecture(top1_acc, top5_acc)
            elif strategy == "finished":
                break
            else:
                raise TypeError("strategy must be 'prune' or 'quantize' or 'finished' ")

            # save the model when training end
            if epoch == settings.DYNAMIC_EPOCH:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, 'models/VGG_Compressed_{:d}.pkl'.format(current_time))
    
    end_time = time.time()
    logging.info('Compression process takes {} minutes'.format((end_time - start_time) / 60))
