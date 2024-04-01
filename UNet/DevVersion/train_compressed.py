import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import copy
import math
from thop import profile
import torch.nn.functional as F

from conf import settings
from utils import get_Carvana_training_validation_dataloader, dice_loss, multiclass_dice_coeff, WarmUpLR
from models.UNet import UNet

def train(epoch):
    net.train()
    with tqdm(total=len(Carvana_training_loader), desc=f'Epoch {epoch}/{settings.DYNAMIC_EPOCH}', unit='img') as pbar:
            for batch_idx, batch in enumerate(Carvana_training_loader):
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                masks_pred = net(images)
                loss = loss_function(masks_pred, true_masks)
                loss += torch.log(dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, net.num_class).permute(0, 3, 1, 2).float(),
                    multiclass=True
                ))

                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                if epoch <= warm:
                    warmup_scheduler.step()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

@torch.inference_mode()
def eval_training(epoch=0, tb=True):
    net.eval()
    dice_score = 0

    # iterate over the validation set
    with torch.cuda.amp.autocast(enabled=amp):
        for batch in tqdm(Carvana_validation_loader, total=len(Carvana_validation_loader), desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.num_class).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_class).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
    val_score = (dice_score / max(len(Carvana_validation_loader), 1)).item()
    logging.info('Validation Dice score: {}'.format(val_score))
    return val_score


def generate_architecture(model, local_top1_accuracy):
    # initialize all evaluating variables
    model_list = []
    top1_accuracy_list = []
    FLOPs_list = []
    parameter_num_list = []
    model_list.append(model)
    top1_accuracy_list.append(local_top1_accuracy)
    local_FLOPs, local_parameter_num = profile(model, inputs = (input, ), verbose=False)
    FLOPs_list.append(local_FLOPs)
    parameter_num_list.append(local_parameter_num)

    original_model = copy.deepcopy(model)
    for model_id in range(1, generate_num + 1):
        with tqdm(total=generate_num, desc=f'Generated architectures', unit='model') as pbar1:
            # generate architecture
            dev_model = copy.deepcopy(original_model)
            UNet.update_architecture(dev_model, modification_num)
            dev_model = dev_model.to(device)
            dev_lr = lr
            dev_optimizer = optim.RMSprop(dev_model.parameters(), lr=current_lr, weight_decay=1e-8, momentum=0.999, foreach=True)
            dev_grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
            dev_warmup_scheduler = WarmUpLR(dev_optimizer, iter_per_epoch * warm)
            dev_top1_accuracies = []
            # train the architecture for dev_num times
            for dev_id in range(1, dev_num + 1):
                with tqdm(total=len(Carvana_training_loader), desc=f'Dev Epoch {dev_id}/{dev_num} for architecture {model_id}', unit='img') as pbar2:
                    if dev_id in settings.DYNAMIC_MILESTONES:
                        dev_lr *= gamma
                        for param_group in dev_optimizer.param_groups:
                            param_group['lr'] = dev_lr
                    # begin training
                    dev_model.train()               # set model into training
                    for batch_idx, batch in enumerate(Carvana_training_loader):
                        # move train data to device
                        images, true_masks = batch['image'], batch['mask']

                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        true_masks = true_masks.to(device=device, dtype=torch.long)
                        # clear the gradient data and update parameters based on error
                        dev_optimizer.zero_grad(set_to_none=True)
                        # get predict y and compute the error
                        masks_pred = dev_model(images)
                        loss = loss_function(masks_pred, true_masks)
                        loss += torch.log(dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, dev_model.num_class).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        ))
                        dev_grad_scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(dev_model.parameters(), max_norm=1.0)
                        dev_grad_scaler.step(dev_optimizer)
                        dev_grad_scaler.update()

                        if dev_id <= warm:
                            dev_warmup_scheduler.step()
                        
                        pbar2.update(images.shape[0])
                        pbar2.set_postfix(**{'loss (batch)': loss.item()})
                
                # discard the first half data as model need retraining
                #if (dev_id + 1) % dev_num >= math.ceil(dev_num / 2):
                # initialize the testing parameters
                dice_score = 0.0
                # begin testing
                dev_model.eval()
                with torch.inference_mode():
                    for batch in Carvana_validation_loader:
                        image, mask_true = batch['image'], batch['mask']
                        # move images and labels to correct device and type
                        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        mask_true = mask_true.to(device=device, dtype=torch.long)
                        # get predict y and predict its class
                        mask_pred = dev_model(image)
                        # convert to one-hot format
                        mask_true = F.one_hot(mask_true, dev_model.num_class).permute(0, 3, 1, 2).float()
                        mask_pred = F.one_hot(mask_pred.argmax(dim=1), dev_model.num_class).permute(0, 3, 1, 2).float()
                        # compute the Dice score, ignoring background
                        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                    # calculate the accuracy and print it
                    top1_accuracy = (dice_score / max(len(Carvana_validation_loader), 1)).item()
                    dev_top1_accuracies.append(top1_accuracy)
                    logging.info('Dev Validation Dice score: {}'.format(top1_accuracy))
            pbar1.update(1)
        # store the model and score
        model_list.append(dev_model)
        top1_accuracy_list.append(dev_top1_accuracies)
        dev_FLOPs, dev_parameter_num = profile(dev_model, inputs = (input, ), verbose=False)
        FLOPs_list.append(dev_FLOPs)
        parameter_num_list.append(dev_parameter_num)
    global Para_compressed_ratio
    score_list = compute_score(model_list, top1_accuracy_list, FLOPs_list, parameter_num_list)
    best_model_index = np.argmax(score_list)
    best_model_FLOPs = FLOPs_list[best_model_index]
    best_model_Params = parameter_num_list[best_model_index]
    FLOPs_compressed_ratio = best_model_FLOPs / original_FLOPs_num
    Para_compressed_ratio = best_model_Params / original_para_num
    model = copy.deepcopy(model_list[best_model_index])
    logging.info('Model {} wins'.format(best_model_index))
    logging.info('Current compression ratio: FLOPs: {}, Parameter number {}'.format(FLOPs_compressed_ratio, Para_compressed_ratio))
    return model, best_model_index


def compute_score(model_list, top1_accuracy_list, FLOPs_list, parameter_num_list):
    score_list = []
    # extract the last element (converged) accuracy to denote that architecture's accuracy
    top1_accuracies = [sublist[-1] for sublist in top1_accuracy_list]
    # use Min-Max Normalization to process the FLOPs_list and parameter_num_list
    FLOPs_tensor = torch.tensor(FLOPs_list)
    parameter_num_tensor = torch.tensor(parameter_num_list)
    FLOPs_scaled = (FLOPs_tensor - torch.min(FLOPs_tensor)) / (torch.max(FLOPs_tensor) - torch.min(FLOPs_tensor))
    parameter_num_scaled = (parameter_num_tensor - torch.min(parameter_num_tensor)) / (torch.max(parameter_num_tensor) - torch.min(parameter_num_tensor))
    for model_id in range(len(model_list)):
        top1_accuracy = top1_accuracies[model_id]
        if np.max(top1_accuracies) > accuracy_threshold:
            # if there exists architecture that is higher than accuracy_threshold, only pick the simplest one and discard other
            if (top1_accuracy > accuracy_threshold - 0.005):
                score_list.append(top1_accuracy * 0.5 +  0.5 - FLOPs_scaled[model_id].item() * 0.25 - parameter_num_scaled[model_id].item() * 0.25)
            else:
                score_list.append(0)
        else:
            score_list.append(top1_accuracy * 1)
    logging.info('Generated Model Top1 Accuracy List: {}'.format(top1_accuracy_list))
    logging.info('Generated Model Parameter Number Scaled List: {}'.format(parameter_num_scaled))
    logging.info('Generated Model Score List: {}'.format(score_list))
    return score_list

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

    args = parser.parse_args()
    check_args(args)

    return args


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    logging.info('Start with random seed: {}'.format(current_time))

    # move the LeNet Module into the corresponding device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-5
    gamma = 0.2
    current_lr = lr * gamma * gamma * gamma
    warm = 1
    batch_size = 1
    amp = True
    generate_num = settings.MAX_GENERATE_NUM
    modification_num = settings.MAX_MODIFICATION_NUM
    tolerance_times = settings.MAX_TOLERANCE_TIMES
    dev_num = settings.DEV_NUM
    if args.criteria == 'accuracy':
        accuracy_threshold = args.accuracy_threshold
        compression_threshold = settings.DEFAULT_COMPRESSION_THRESHOLD
    else:
        accuracy_threshold = settings.DEFAULT_ACCURACY_THRESHOLD
        compression_threshold = args.compression_threshold

    # data preprocessing:
    Carvana_training_loader, Carvana_validation_loader = get_Carvana_training_validation_dataloader(
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    net = torch.load('models/UNet_Compressed_1711942631.pkl')  # replace it with the model gained by train_original.py
    net = net.to(device).to(memory_format=torch.channels_last)

    input = torch.rand(1, 3, 640, 959).to(device)
    original_FLOPs_num, original_para_num = profile(net, inputs = (input, ), verbose=False)
    Para_compressed_ratio = 1.000

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=current_lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    iter_per_epoch = len(Carvana_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    
    for epoch in range(1, settings.DYNAMIC_EPOCH + 1):
        #train(epoch)
        #val_score = eval_training(epoch)
        val_score = 0

        # dynamic generate architecture
        if epoch % 2 == 0:
            if tolerance_times >= 0:
                net, model_index = copy.deepcopy(generate_architecture(net, [val_score]))
                net = net.to(device).to(memory_format=torch.channels_last)
                if model_index == 0:
                    tolerance_times -= 1
                    modification_num /= 2
                    generate_num += 1
                optimizer = optim.RMSprop(net.parameters(), lr=current_lr, weight_decay=1e-8, momentum=0.999, foreach=True)
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, 'models/UNet_Compressed_{:d}.pkl'.format(current_time))
            else:
                if args.criteria == 'compression' and Para_compressed_ratio >= compression_threshold:
                    # decrement accuracy_threshold and reinitialize hyperparameter if criteria is compression ratio
                    accuracy_threshold -= 0.05
                    generate_num = settings.MAX_GENERATE_NUM
                    modification_num = settings.MAX_MODIFICATION_NUM
                    tolerance_times = settings.MAX_TOLERANCE_TIMES
                    logging.info('Current accuracy threshold: {}'.format(accuracy_threshold))
                else:
                    # save the module and break
                    if not os.path.isdir("models"):
                        os.mkdir("models")
                    torch.save(net, 'models/UNet_Compressed_{:d}.pkl'.format(current_time))
                    break

            # save the model when training end
            if epoch == settings.DYNAMIC_EPOCH:
                # save the module
                if not os.path.isdir("models"):
                    os.mkdir("models")
                torch.save(net, 'models/UNet_Compressed_{:d}.pkl'.format(current_time))
