## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from Denoising.Restormer.basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx


def mot_deblur(image, model_restoration):
  factor = 8
  img = np.float32(image)/255.
  img = torch.from_numpy(img).permute(2,0,1)
  input_ = img.unsqueeze(0).cuda()

  # Padding in case images are not multiples of 8
  h,w = input_.shape[2], input_.shape[3]
  H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
  padh = H-h if h%factor!=0 else 0
  padw = W-w if w%factor!=0 else 0
  input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

  restored = model_restoration(input_)

  # Unpad images to original dimensions
  restored = restored[:,:,:h,:w]
  restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

  return restored

