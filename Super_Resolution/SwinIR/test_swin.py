import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from Super_Resolution.SwinIR.models.network_swinir import SwinIR as net
from Super_Resolution.SwinIR.utils import util_calculate_psnr_ssim as util


def test_swinir(model, img_lq, device):
  window_size = 8
  scale = 4
  #imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
  # img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
  # img_lq = torch.from_numpy(img_lq/255.).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
  print(img_lq.shape)
  img_lq = torch.from_numpy(np.transpose((img_lq),(0,3,1,2))/255.).float().to(device)
  print(img_lq.shape)
  # inference
  with torch.no_grad():
      # pad input image to be a multiple of window_size
      _, _, h_old, w_old = img_lq.size()
      h_pad = (h_old // window_size + 1) * window_size - h_old
      w_pad = (w_old // window_size + 1) * window_size - w_old
      img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
      img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
      #print(img_lq.shape)
      output = model(img_lq)
      #print(output.shape)
      output = output[..., :h_old * scale, :w_old * scale]
      #print(output.shape)
      output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
      if output.ndim == 3:
          output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
      output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
      print(output.shape)
      return np.transpose(output,(0,2,3,1))



def define_model(model_path, device, large_model=True):
  
  #model = define_model(args)

  if not large_model:
      # use 'nearest+conv' to avoid block artifacts
      model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                  img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                  mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
  else:
      # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
      model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                  img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                  num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                  mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
  param_key_g = 'params_ema'
  print("udfhvbdfuvydb")
  pretrained_model = torch.load(model_path)
  model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
  
  model.eval()
  model = model.to(device)
  
  return model