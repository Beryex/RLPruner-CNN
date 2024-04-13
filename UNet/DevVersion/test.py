import torch
from thop import profile
import logging
from PIL import Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from conf import settings
from utils import BasicDataset, get_Carvana_training_validation_dataloader, dice_loss, multiclass_dice_coeff, WarmUpLR

def predict():
    in_file = '0cdf5b5d0ce1_14.jpg'
    out_file = '0cdf5b5d0ce1_14_Predict.jpg'

    model = torch.load('models/UNet_Compressed_1712007552.pkl')
    model = model.to(device)
    model.eval()

    Full_img = Image.open(in_file)

    model.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, Full_img, scale=1.0, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(img).cpu()
        output = F.interpolate(output, (Full_img.size[1], Full_img.size[0]), mode='bilinear')
        mask = output.argmax(dim=1)

    mask = mask[0].long().squeeze().numpy()
    mask_values = [0, 1]

    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    
    result = Image.fromarray(out)
    result.save(out_file)

def test():
    Carvana_training_loader, Carvana_validation_loader = get_Carvana_training_validation_dataloader(
        num_workers=4,
        batch_size=1,
        shuffle=True
    )

    original_para_num = 0.0
    original_FLOPs_num = 0.0
    compressed_para_num = 0.0
    compressed_FLOPs_num = 0.0
    input = torch.rand(1, 3, 640, 959).to(device)

    # initialize the testing parameters
    dice_score = 0.0

    # begin testing
    model = torch.load('models/UNet_Original_1711995741.pkl')
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(Carvana_validation_loader, total=len(Carvana_validation_loader), desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            # predict the mask
            mask_pred = model(image)
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, model.num_class).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.num_class).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
    val_score = (dice_score / max(len(Carvana_validation_loader), 1)).item()
    original_FLOPs_num, original_para_num = profile(model, inputs = (input, ), verbose=False)
    logging.info('Original model has dice score: {}'.format(val_score))
    logging.info('Original model has FLOPs: {}, Parameter Num: {}'.format(original_FLOPs_num, original_para_num))

    # initialize the testing parameters
    dice_score = 0.0

    # begin testing
    model = torch.load('models/UNet_Compressed_1712007552.pkl')
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(Carvana_validation_loader, total=len(Carvana_validation_loader), desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            # predict the mask
            mask_pred = model(image)
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, model.num_class).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.num_class).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
    compressed_FLOPs_num, compressed_para_num = profile(model, inputs = (input, ), verbose=False)
    val_score = (dice_score / max(len(Carvana_validation_loader), 1)).item()
    logging.info('Compressed model has dice score: {}'.format(val_score))
    logging.info('Compressed model has FLOPs: {}, Parameter Num: {}'.format(compressed_FLOPs_num, compressed_para_num))
    # get compressed ratio
    FLOPs_compressed_ratio = compressed_FLOPs_num / original_FLOPs_num
    Para_compressed_ratio = compressed_para_num / original_para_num
    logging.info('FLOPS compressed ratio: {}, Parameter Num compressed ratio: {}'.format(FLOPs_compressed_ratio, Para_compressed_ratio))
    # print model
    print(model)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # move the LeNet Module into the corresponding device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict()
    test()
    