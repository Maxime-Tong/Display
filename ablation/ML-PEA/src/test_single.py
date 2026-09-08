import torch
import numpy as np
from util import (
    load_checkpoint,
    to_np_8b
)
from unet import UNet
from test import compute_model
from options import Options
import imageio.v3 as iio

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = Options().parse()

# model init
model = UNet(3, opt.channels)
model.to(device)
ckpt_path = "ckpts/R0.68/latest.ckpt"
state = "ckpts/R0.68/state.txt"
_, model = load_checkpoint(model, ckpt_path=ckpt_path, state=state)

# Run inference
img = iio.imread('../assets/bird.jpg')/255
input = torch.from_numpy(img[np.newaxis,...]).permute(0,3,1,2)[:,:3,...].to(device).float() # [1x3xHxW]
filter_mask, output = compute_model(model, opt, input)

# plot result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(to_np_8b(input))
ax1.set_title('Reference')
ax1.axis('off')

ax2.imshow(to_np_8b(output))
ax2.set_title('ML-PEA')
ax2.axis('off')

plt.show()