import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import SSIM
from glob import glob

from data_loader import ImageDataset
from options import Options
from util import (
    load_checkpoint,
    srgb2rgb,
    rgb2srgb,
    PSNR,
    to_np_8b,
)
from vgg import VGGLoss
from power_loss import OLED_loss, OLED_Model
from unet import UNet

import imageio.v3 as iio
import pandas as pd
import lpips

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn_alex = lpips.LPIPS(net='alex')

def init_losses(opt):
    # ========================================
    #  initialising losses and optimizer
    # ========================================
    l1 = torch.nn.L1Loss()
    l2 = torch.nn.MSELoss()
    perceptual_loss = VGGLoss(device=device)
    ssim_module = SSIM(data_range=1.0).to(device)
    power = OLED_loss

    loss_dict = {
        "L1": {
            "fn": l1,
            "w": opt.w_l1
        },
        "L2": {
            "fn": l2,
            "w": opt.w_l2
        },
        "VGG": {
            "fn": perceptual_loss,
            "w": opt.w_vgg
        },
        "Power": {
            "fn": power,
            "w": opt.w_power
        },
        "SSIM": {
            "fn": ssim_module,
            "w": opt.w_ssim
        }
    }
    return loss_dict

def init_vars():
    opt = Options().parse()
    
    loss_dict = init_losses(opt)
    ### Create path names:
    if opt.pathname == "default":
        folder = "{}_{}_{}_CH{}".format("OLED", opt.r, opt.method, opt.channels)
        for k in loss_dict:
            d_w = loss_dict[k]['w']
            if d_w > 0:
                folder = folder + "_{}_{:.1f}".format(k, d_w)

        nums = [int(x.split("/")[-1]) for x in glob(os.path.join("results", folder, "*"))]
        max_v = 0
        opt.pathname = os.path.join(folder, str(max_v))
        
    dataset_test = ImageDataset(mode="test", opt=opt)
    data_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)

    # model init
    model = UNet(3, opt.channels)
    model.to(device)
        
    os.makedirs("{}/{}/test".format(opt.result_folder, opt.pathname), exist_ok=True)

    _, model = load_checkpoint(model, "{}/{}/checkpoints/latest.ckpt".format(opt.result_folder, opt.pathname), "{}/{}/checkpoints/state.txt".format(opt.result_folder, opt.pathname))

    return data_loader, model, opt

def compute_model(model, opt, input):
    if opt.method == "MULT":
        filter_mask = model(input)
        output = (filter_mask + 1)/2 * input
    elif opt.method == "ADD":
        filter_mask = model(input)
        output = input - (filter_mask + 1)/2
    return filter_mask, torch.clamp(output, 0, 1)

def validate(data_loader_test, model, opt, test_name=""):
    test_dir = "results/{}/test".format(opt.pathname)
    ssim_fn = SSIM()
    os.makedirs(os.path.join(test_dir, "inputs", opt.dataset), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "outputs", opt.dataset), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "uniforms", opt.dataset), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "filters", opt.dataset), exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        gt_power, pred_power, psnr, psnr_uniforms, uniform_power, ssim_uniforms, ssim_preds, lpips_uniforms, lpips_preds = [], [], [], [], [], [], [], [], []
        i = 0
        for _, data in enumerate(tqdm(data_loader_test, desc="Validation %")):
            
            input = data["img"].data.to(device)
            
            filter_mask, output = compute_model(model, opt, input)

            power_model = OLED_Model
            
            gt_power_temp = power_model(input)[1]
            pred_power_temp = power_model(output)[1]
            relative_power_temp = (1 - pred_power_temp / gt_power_temp) * 100
            
            uniform_img = rgb2srgb(srgb2rgb(input) * (1-relative_power_temp/100))
            uniform_power_temp = power_model(uniform_img)[1]
            
            gt_power += [gt_power_temp.item()]
            pred_power += [pred_power_temp.item()]
            uniform_power += [uniform_power_temp.item()]

            psnr_uniform = PSNR(input, uniform_img, 1)
            psnr_uniforms += [psnr_uniform.item()]
            
            psnr_val = PSNR(input, output, 1)
            if type(psnr_val) == int:
                psnr += [psnr_val]
            else:
                psnr += [psnr_val.item()]
                
            ssim_uniform = ssim_fn(input, uniform_img)
            ssim_uniforms += [ssim_uniform.item()]
            ssim_pred = ssim_fn(input, output)
            ssim_preds += [ssim_pred.item()]
            
            lpips_uniform = loss_fn_alex(input, uniform_img)
            lpips_uniforms += [lpips_uniform.item()]
            lpips_pred = loss_fn_alex(input, output)
            lpips_preds += [lpips_pred.item()]

            """
            Save generated images
            """
            input = to_np_8b(input)
            output = to_np_8b(output)
            uniform_img = to_np_8b(uniform_img)
            filter_mask = to_np_8b(filter_mask)
            
            if filter_mask.shape[2] == 1:
                filter_mask = filter_mask[...,0]

            pth = data['path'][0].split("/")[-1]
            iio.imwrite(os.path.join(test_dir, "inputs", opt.dataset, pth), input)
            iio.imwrite(os.path.join(test_dir, "outputs", opt.dataset, pth), output)
            iio.imwrite(os.path.join(test_dir, "uniforms", opt.dataset, pth), uniform_img)
            iio.imwrite(os.path.join(test_dir, "filters", opt.dataset, pth), filter_mask)
                        
            i += 1
            
        # Write metrics:
        data = {'psnr_uniform': psnr_uniforms, 'psnr_pred': psnr, 'ssim_uniform': ssim_uniforms, 'ssim_pred': ssim_preds, 'lpips_uniform': lpips_uniforms, 'lpips_pred': lpips_preds}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(test_dir, 'outputs.csv'), index=False)
        
if __name__ == "__main__":
    data_loader, model, opt = init_vars()
    validate(data_loader, model, opt)