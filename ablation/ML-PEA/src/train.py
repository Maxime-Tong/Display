import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob

from pytorch_msssim import SSIM
from vgg import VGGLoss
from power_loss import OLED_loss, OLED_Model

from unet import UNet

from data_loader import ImageDataset
from options import Options
from util import (
    make_directories,
    save_checkpoint,
    load_checkpoint,
    srgb2rgb,
    rgb2srgb,
    to_np_8b,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
def init_losses(opt):
    # initialising losses and optimizer
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
    # initialise training options
    opt = Options().parse()
        
    loss_dict = init_losses(opt)
    
    # create path names:
    if opt.pathname == "default":
        folder = "{}_{}_{}_CH{}".format("OLED", opt.r, opt.method, opt.channels)
        for k in loss_dict:
            d_w = loss_dict[k]['w']
            if d_w > 0:
                folder = folder + "_{}_{:.1f}".format(k, d_w)

        nums = [int(x.split("/")[-1]) for x in glob(os.path.join("results", folder, "*"))]
        if len(nums) == 0:
            max_v = 0
        else:
            max_v = np.asarray(nums).max() + 1
        opt.pathname = os.path.join(folder, str(max_v))

    # loading data
    dataset = ImageDataset(mode="train", dim=(256,256), opt=opt)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    dataset_test = ImageDataset(mode="test", opt=opt)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    # model init
    model = UNet(3, opt.channels)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    make_directories(pathname=opt.pathname)
    
    with open("{}/{}/commandline_args.txt".format("results", opt.pathname), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    if opt.savedir != "":
        start_epoch, model = load_checkpoint(model, "{}/checkpoints/latest.ckpt".format(opt.savedir), "{}/checkpoints/state.txt".format(opt.savedir))
    else:
        start_epoch = 1

    return data_loader, data_loader_test, model, loss_dict, optimizer, start_epoch, opt

def compute_model(model, opt, input):
    if opt.method == "MULT":
        dimming_map = model(input)
        output = (dimming_map + 1)/2 * input
    elif opt.method == "ADD":
        dimming_map = model(input)
        output = input - (dimming_map + 1)/2
    return dimming_map, torch.clamp(output, 0, 1)

def train(data_loader, data_loader_test, model, loss_dict, optimizer, start_epoch, opt):
    loss_pth = "{}/{}/losses/".format("results", opt.pathname)
    if not os.path.exists(loss_pth):
        os.makedirs(loss_pth)
        
    ### Init running loss:
    running_loss = {}
    for k in loss_dict:
        running_loss[k] = []
        
    ### Training loop:
    for epoch in range(start_epoch, opt.epochs + 1):
        
        # Reset loss dictionary
        loss_dict_temp = {}
        for k in loss_dict:
            loss_dict_temp[k] = []

        for _, data in enumerate(tqdm(data_loader, desc="Batch %")):

            optimizer.zero_grad()

            input = data["img"].data.to(device)
            _, output = compute_model(model, opt, input)
            
            # loss function
            loss = 0
            for k in loss_dict:
                
                w = loss_dict[k]["w"]
                if w == 0:
                    continue
                if k == "Power":
                    loss_temp = loss_dict[k]["fn"](output, input, opt=opt)
                elif k == "SSIM":
                    loss_temp = 1-loss_dict[k]["fn"](input, output)
                else:
                    loss_temp = loss_dict[k]["fn"](output, input)
                
                loss_mean = torch.mean(loss_temp)
                loss += w * loss_mean
                loss_dict_temp[k] += [loss_mean.item()]

            # backpropagate and step
            loss.backward()
            optimizer.step()
                    
        ### Run validation set
        with torch.no_grad():
            for k in loss_dict:
                if loss_dict[k]["w"] == 0:
                    continue
                running_loss[k] += [np.mean(loss_dict_temp[k])]
                plt.plot(np.arange(len(running_loss[k])), running_loss[k], label=k)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(loss_pth + "{}.png".format(k), bbox_inches="tight", dpi=300)
                plt.close()

            if epoch % opt.save_results_after == 0:  # save image results
                validation(data_loader_test, model, epoch, opt)

            if epoch % opt.save_ckpt_after == 0:
                save_checkpoint(epoch, model, opt.pathname)

def validation(data_loader_test, model, epoch, opt, test_name=""):
    if len(test_name) == 0:
        test_pth = "{}/{}/test_results".format("results", opt.pathname)
        if not os.path.exists(test_pth):
            os.makedirs(test_pth)
    else:
        test_pth = "{}/{}/{}/test_results".format("results", opt.pathname, test_name)
        if not os.path.exists(test_pth):
            os.makedirs(test_pth)
    
    model.eval()
    with torch.no_grad():
        gt_power, pred_power, uniform_power = [], [], []
        i = 0
        for _, data in enumerate(tqdm(data_loader_test, desc="Validation %")):
            
            input = data["img"].data.to(device)
            dimming_map, output = compute_model(model, opt, input)

            power_model = OLED_Model
            
            gt_power_temp = power_model(input)[1]
            pred_power_temp = power_model(output)[1]
            relative_power_temp = (1 - pred_power_temp / gt_power_temp) * 100
            
            uniform_img = rgb2srgb(srgb2rgb(input) * (1-relative_power_temp/100))
            uniform_power_temp = power_model(uniform_img)[1]
            relative_power_temp_uniform = (1 - uniform_power_temp / gt_power_temp) * 100
            
            gt_power += [gt_power_temp.item()]
            pred_power += [pred_power_temp.item()]
            uniform_power += [uniform_power_temp.item()]

            # Plot generated images
            input = to_np_8b(input)
            output = to_np_8b(output)
            uniform_img = to_np_8b(uniform_img)
            dimming_map = to_np_8b(dimming_map)

            fig, (axs) = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
            axs[0,0].imshow(input)
            axs[1,1].imshow(dimming_map)
            axs[1,0].imshow(output)
            axs[0,1].imshow(uniform_img)
            axs[0,0].title.set_text("GT")
            axs[1,1].title.set_text("Filter")
            axs[1,0].title.set_text("PRED, {:.1f})".format(relative_power_temp))
            axs[0,1].title.set_text("UNIFORM, {:.1f})".format(relative_power_temp_uniform))
            fig.subplots_adjust(wspace=0)
            plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False) 

            plt.axis('off')
            plt.savefig("{}/test_{}.png".format(test_pth, i), bbox_inches="tight", dpi=300)
            plt.close()
            
            i += 1

if __name__ == "__main__":
    data_loader, data_loader_test, model, loss_dict, optimizer, start_epoch, opt = init_vars()
    train(data_loader, data_loader_test, model, loss_dict, optimizer, start_epoch, opt)