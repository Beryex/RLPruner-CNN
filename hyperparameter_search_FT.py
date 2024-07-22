import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import wandb
import logging

from conf import settings
from utils import (extract_prunable_layers_info, extract_prunable_layer_dependence, 
                   adjust_prune_distribution_for_cluster, get_dataloader, setup_logging, 
                   Prune_agent, torch_set_random_seed, torch_resume_random_seed)


train_loader, valid_loader, test_loader, _, _ = get_dataloader('cifar100', 
                                                                batch_size=256, 
                                                                num_workers=8)
eval_loader = test_loader
device = 'cuda'
loss_function = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model: nn.Module):
    """ Evaluate model on eval_loader """
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    eval_loss = 0.0
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        eval_loss += loss_function(outputs, labels).item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(eval_loader.dataset)
    top5_acc = correct_5 / len(eval_loader.dataset)
    eval_loss /= len(eval_loader)
    return top1_acc, top5_acc, eval_loss


def fine_tuning_with_KD(teacher_model: nn.Module,
                        student_model: nn.Module,
                        optimizer: optim.Optimizer, 
                        T: float = 2,
                        soft_loss_weight: float = 0.25, 
                        stu_loss_weight: float = 0.75) -> float:
    """ fine tuning generated model with knowledge distillation with original model as teach """
    teacher_model.eval()
    student_model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        stu_outputs = student_model(images)
        stu_loss = loss_function(stu_outputs, labels)

        with torch.no_grad():
            tch_outputs = teacher_model(images)
        # soften the student output by applying softmax firstly and log() 
        # secondly to avoid overflow and improve efficiency, and teacher output softmax only
        stu_outputs_softened = nn.functional.log_softmax(stu_outputs / T, dim=-1)
        tch_outputs_softened = nn.functional.softmax(tch_outputs / T, dim=-1)
        soft_loss = (torch.sum(tch_outputs_softened * (tch_outputs_softened.log() - stu_outputs_softened)) 
                     / stu_outputs_softened.shape[0] * T ** 2)

        loss = soft_loss_weight * soft_loss + stu_loss_weight * stu_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss

teacher_model = torch.load("models/vgg16_cifar100_1721008389_original.pth")
generated_model = torch.load("models/1721025351_checkpoint.pth")

setup_logging(experiment_id=int(time.time()), 
            model_name='vgg16', 
            dataset_name='cifar100', 
            action='test')
"""
epoch_total = 20
initial_lr=5 * 10 ** -3
min_lr = 1 * 10 ** -6
stu_co = 0.25

teacher_model = torch.load("models/vgg16_cifar100_1721008389_original.pth")
generated_model = torch.load("models/1721025351_checkpoint.pth")
torch_set_random_seed(1)
logging.info(f'\n')
optimizer = optim.SGD(generated_model.parameters(),lr=initial_lr, 
        momentum=0.9,weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        epoch_total,eta_min=min_lr,last_epoch=-1)
best_acc = -1
best_trained_generated_model_with_info = None
for dev_epoch in range(1, epoch_total + 1):
    train_loss = fine_tuning_with_KD(teacher_model=teacher_model,
            student_model=generated_model,
            optimizer=optimizer,
            soft_loss_weight= 1 - stu_co,
            stu_loss_weight=stu_co)
    top1_acc, top5_acc, _ = evaluate(generated_model)
    lr_scheduler.step()
    logging.info(f'train_loss:{train_loss}, top1_acc: {top1_acc}')
    if best_acc < top1_acc:
        best_acc = top1_acc
print(f'best acc: {best_acc}, epoch_total: {epoch_total}, initial_lr: {initial_lr}, min_lr: {min_lr}, stu_co: {stu_co}')
logging.info(f'best acc: {best_acc}, epoch_total: {epoch_total}, initial_lr: {initial_lr}, min_lr: {min_lr}, stu_co: {stu_co}')
wandb.log({"best acc": best_acc, "epoch_total": epoch_total, "initial_lr": initial_lr, "min_lr": min_lr, "stu_co": stu_co})
import sys
sys.exit()
"""
for epoch_total in np.arange(1, 30, 3):
    for initial_lr_ratio in np.arange(-2.5, -3.5 - 0.1, -0.5):
        initial_lr = 10 ** initial_lr_ratio
        for min_lr_ratio in np.arange(-4.5, -5.5 - 0.1, -0.5):
            min_lr = 10 ** min_lr_ratio
            for stu_co in np.arange(0, 1 + 0.1, 0.1):
                for i in range(2):
                    if i == 1:
                        actual_epoch = epoch_total + 5
                    else:
                        actual_epoch = epoch_total
                    teacher_model = torch.load("models/vgg16_cifar100_1721008389_original.pth")
                    generated_model = torch.load("models/test.pth")
                    torch_set_random_seed(1)
                    logging.info(f'\n')
                    optimizer = optim.SGD(generated_model.parameters(), 
                                        lr=initial_lr, momentum=0.9, 
                                        weight_decay=5e-4)
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                    epoch_total, 
                                                                    eta_min=min_lr,
                                                                    last_epoch=-1)
                    best_acc = -1
                    best_trained_generated_model_with_info = None
                    for dev_epoch in range(1, actual_epoch + 1):
                        train_loss = fine_tuning_with_KD(teacher_model=teacher_model, 
                                                    student_model=generated_model, 
                                                    optimizer=optimizer,
                                                    soft_loss_weight= 1 - stu_co,
                                                    stu_loss_weight=stu_co)
                        top1_acc, top5_acc, _ = evaluate(generated_model)
                        lr_scheduler.step()
                        logging.info(f'train_loss:{train_loss}, top1_acc: {top1_acc}')

                        if best_acc < top1_acc:
                            best_acc = top1_acc
                    print(f'best acc: {best_acc}, epoch_total: {epoch_total}, initial_lr: {initial_lr}, min_lr: {min_lr}, stu_co: {stu_co}, use_addi: {i}')
                    logging.info(f'best acc: {best_acc}, epoch_total: {epoch_total}, initial_lr: {initial_lr}, min_lr: {min_lr}, stu_co: {stu_co}, use_addi: {i}')
                    wandb.log({"best acc": best_acc, "epoch_total": epoch_total, "initial_lr": initial_lr, "min_lr": min_lr, "stu_co": stu_co, "use_addi": i})
            if epoch_total == 1:
                break