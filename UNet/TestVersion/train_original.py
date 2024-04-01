import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import logging

from conf import settings
from utils import get_Carvana_training_validation_dataloader, dice_loss, multiclass_dice_coeff, WarmUpLR
from models.UNet import UNet

def train(epoch):
    net.train()
    with tqdm(total=len(Carvana_training_loader), desc=f'Epoch {epoch}/{settings.ORIGINAL_EPOCH}', unit='img') as pbar:
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
def eval_training(epoch):
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # reinitialize random seed
    current_time = int(time.time())
    torch.manual_seed(current_time)
    logging.info('Start with random seed: {}'.format(current_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-5
    gamma = 0.2
    current_lr = lr
    warm = 1
    batch_size = 1
    amp = True

    # data preprocessing:
    Carvana_training_loader, Carvana_validation_loader = get_Carvana_training_validation_dataloader(
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    net = UNet(num_class=2).to(device).to(memory_format=torch.channels_last)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=current_lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    iter_per_epoch = len(Carvana_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    best_score = 0.0

    for epoch in range(1, settings.ORIGINAL_EPOCH + 1):
        if epoch in settings.ORIGINAL_MILESTONES:
            current_lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        train(epoch)
        val_score = eval_training(epoch)
        
        #start to save best performance model after second milestone
        if epoch > settings.ORIGINAL_MILESTONES[0] and best_score < val_score:
            best_score = val_score
            # save the module
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, 'models/UNet_Original_{:d}.pkl'.format(current_time))
