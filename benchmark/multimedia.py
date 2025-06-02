import cv2
from PIL import Image
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

'''==========function for superresolution=========='''
def process_with_sr_model(image_path, sr_model, device='cuda'):
    # Load and transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Error handling for file loading
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    except Exception as e:
        raise Exception(f"Error loading image: {e}")
    
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Ensure model is on the same device
    sr_model = sr_model.to(device)
    sr_model.eval()
    
    with torch.no_grad():
        output_tensor = sr_model(image_tensor)
    
    # Convert back to image
    def tensor_to_image(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Move tensor to CPU first to avoid device issues
        tensor = tensor.cpu()
        
        # Create mean and std tensors on CPU
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Denormalize
        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)
        
        # Convert to numpy and PIL
        np_img = (tensor.mul(255).byte().permute(1, 2, 0).numpy())
        return Image.fromarray(np_img)
    
    result_image = tensor_to_image(output_tensor)
    
    return result_image


'''==========import from our code=========='''
sys.path.append('.')
from torchvision import transforms
import config as cfg
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab
from sr_model import CT_CA_skip_Model

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

sr_model = CT_CA_skip_Model(num_channels=3).to(torch.device("cuda"))
checkpoint_sr = torch.load('ckpt/channel_skip_model_epoch_30.pth', map_location='cpu')
sr_model.load_state_dict(checkpoint_sr['model_state_dict'])



print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K   Model: {model.name}   TTA: {TTA}')
path = args.path
f = open(path + 'tri_trainlist_complete.txt', 'r')
psnr_list, ssim_list = [], []
for i in f:
    name = str(i).strip().split(',')[0]
    if(len(name) <= 1):
        continue
    I0 = cv2.imread(path + name + '/im1.png')
    I1 = cv2.imread(path + name + '/im2.png')
    I2 = cv2.imread(path + name + '/im3.png') # BGR -> RBG
    # I3 = cv2.imread(path + name + '/im4.png')
    I4 = cv2.imread(path + name + '/im5.png')
    I5 = cv2.imread(path + name + '/im6.png')
    I6 = cv2.imread(path + name + '/im7.png')

    I0 = cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    # I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2RGB)
    I4 = cv2.cvtColor(I4, cv2.COLOR_BGR2RGB)
    I5 = cv2.cvtColor(I5, cv2.COLOR_BGR2RGB)
    I6 = cv2.cvtColor(I6, cv2.COLOR_BGR2RGB)

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
    # I2 = cv2.resize(I2, I3.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    # I4 = cv2.resize(I4, I3.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I4 = (torch.tensor(I4.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    # I2 = sr_model(I2)
    # I4 = sr_model(I4)
    mid = model.inference(I2, I4, TTA=TTA, fast_TTA=TTA)[0].unsqueeze(0)[0]

    mid = mid.detach().cpu().numpy().transpose(1, 2, 0)
    mid = cv2.cvtColor(mid, cv2.COLOR_RGB2BGR)
    mid = (mid * 255).astype(np.uint8)
    name_underline = name.replace('/', '_')
    cv2.imwrite(f'test_result/test_{name_underline}.jpg', mid)
    mid = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)

    # spatio superresolution start!!
    result_img = process_with_sr_model(f'test_result/test_{name_underline}.jpg', sr_model=sr_model, device='cuda')
    # mid = Image.open(f'test_result/test_{name_underline}.jpg')
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std),
    # ])
    # mid = transform(mid).unsqueeze(0).cuda()
    # mid = sr_model(mid).squeeze(0)
    # pil_mid = mid.detach().cpu()
    # for i in range(3):
    #     pil_mid[i] = pil_mid[i] * std[i] + mean[i]
    # pil_mid = torch.clamp(pil_mid, 0, 1)
    # to_pil = transforms.ToPILImage()
    # pil_mid = to_pil(pil_mid)

    result_img.save(f'tmp2.jpg')


    # mid = cv2.imread(f'test_result/test_{name_underline}.jpg')
    mid = cv2.imread('tmp2.jpg')
    mid = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)
    mid = torch.tensor(mid.transpose(2, 0, 1)).cuda() / 255.

    # mid = sr_model(mid.unsqueeze(0))
    # mid = mid.squeeze(0)


    # mid = torch.nn.functional.interpolate(
    #     mid,
    #     size=(256, 448),
    #     mode='bilinear',
    #     align_corners=False
    # )[0]

    # ssim = ssim_matlab(torch.tensor(I3.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
    # mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
    # mid = cv2.cvtColor(mid, cv2.COLOR_RGB2BGR)
    # # cv2.imwrite(f'test_result/test_{name_underline}.jpg', (mid * 255).astype(np.uint8))
    # mid = cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)
    # I3 = I3 / 255.
    # psnr = -10 * math.log10(((I3 - mid) * (I3 - mid)).mean())
    # psnr_list.append(psnr)
    # ssim_list.append(ssim)


    # print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
