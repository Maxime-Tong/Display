import torch

from util import srgb2rgb

def OLED_Model(img):
    H, W = img.shape[2:]
    img_rgb = srgb2rgb(img)
    r = torch.sum(img_rgb[:,0,...])* 231.5384684 / (H*W)
    g = torch.sum(img_rgb[:,1,...]) * 245.6795914 / (H*W)
    b = torch.sum(img_rgb[:,2,...]) * 530.7596369 / (H*W)
    static_power = r/231.5384684 + g/245.6795914 + b/530.7596369 + 977.2813229
    dynamic_power = static_power - 977.2813229
    return static_power, dynamic_power

def OLED_loss(output, input, opt={}):
    return (OLED_Model(output)[1] - opt.r * OLED_Model(input)[1]) ** 2
