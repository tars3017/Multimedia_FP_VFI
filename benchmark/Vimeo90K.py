import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small', 'ours_official'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()


print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K   Model: {model.name}   TTA: {TTA}')
path = args.path
f = open(path + '/tri_trainlist_complete.txt', 'r')
psnr_list, ssim_list = [], []
for i in f:
    name = str(i).strip().split(',')[0]
    if(len(name) <= 1):
        continue
    I0 = cv2.imread(path + name + '/im1.png')
    I1 = cv2.imread(path + name + '/im2.png')
    I2 = cv2.imread(path + name + '/im3.png') # BGR -> RBG
    I3 = cv2.imread(path + name + '/im4.png')
    I4 = cv2.imread(path + name + '/im5.png')
    I5 = cv2.imread(path + name + '/im6.png')
    I6 = cv2.imread(path + name + '/im7.png')

    # Multiple Interpolation
    # I0 = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I1 = (torch.tensor(I1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # # I3 = (torch.tensor(I3.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I4 = (torch.tensor(I4.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I5 = (torch.tensor(I5.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I6 = (torch.tensor(I6.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # mid1 = model.inference(I0, I1, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)
    # mid2 = model.inference(I5, I6, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)
    # mid3 = model.inference(mid1, I2, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)
    # mid4 = model.inference(mid2, I4, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)
    # mid = model.inference(mid3, mid4, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)

    # Single Interpolation
    I2 = cv2.resize(I2, I3.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    I4 = cv2.resize(I4, I3.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I4 = (torch.tensor(I4.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    mid = model.inference(I2, I4, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)[0]


    # mid = torch.nn.functional.interpolate(
    #     mid,
    #     size=(256, 448),
    #     mode='bilinear',
    #     align_corners=False
    # )[0]

    ssim = ssim_matlab(torch.tensor(I3.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
    mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
    name_underline = name.replace('/', '_')
    cv2.imwrite(f'test_result/test_{name_underline}.jpg', (mid * 255).astype(np.uint8))
    I3 = I3 / 255.
    psnr = -10 * math.log10(((I3 - mid) * (I3 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)


    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
