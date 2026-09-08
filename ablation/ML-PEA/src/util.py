import os

import numpy as np
import torch

def PSNR(original, compressed, max_pixel): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0): 
        return 100
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def rgb2srgb(lin):
    lin = torch.clamp(lin, 0.0, 1.0)
    srgb = torch.where(lin > 0.0031308, (1.055 * (lin ** (1/2.4))) - 0.055, 12.92 * lin)
    return srgb

def srgb2rgb(srgb):
    srgb = torch.clamp(srgb, 0.0, 1.0)
    lin = torch.where(srgb > 0.04045, ((srgb + 0.055) / 1.055)**2.4, srgb/12.92)
    return lin

def rgb2srgb_np(lin):
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin > 0.0031308, (1.055 * (lin ** (1/2.4))) - 0.055, 12.92 * lin)
    return srgb

def srgb2rgb_np(srgb):
    srgb = np.clip(srgb, 0.0, 1.0)
    lin = np.where(srgb > 0.04045, ((srgb + 0.055) / 1.055)**2.4, srgb/12.92)
    return lin

def make_directories(pathname):
    dir = "checkpoints"
    if not os.path.exists("{}/{}/{}".format("results", pathname, dir)):
        os.makedirs("{}/{}/{}".format("results", pathname, dir), exist_ok=True)
        
def save_checkpoint(epoch, model, pathname):
    latest_path = os.path.join("{}/{}/checkpoints".format("results", pathname), "latest.ckpt")
    torch.save(model.state_dict(), latest_path)
    np.savetxt("{}/{}/checkpoints/state.txt".format("results", pathname), [epoch + 1], fmt="%d")
    
def load_checkpoint(model, ckpt_path, state):
    start_epoch = np.loadtxt(state, dtype=int)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print("Resuming from epoch ", start_epoch)
    return start_epoch, model
    
def to_np_8b(img):
    return (np.array(img[0].permute(1,2,0).detach().cpu().numpy(), dtype='f') * 255).astype(np.uint8)
