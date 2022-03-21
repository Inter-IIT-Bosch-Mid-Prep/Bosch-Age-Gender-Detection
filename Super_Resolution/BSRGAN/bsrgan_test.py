import os.path
import logging
import numpy as np
import torch
from Super_Resolution.BSRGAN.utils import utils_logger
from Super_Resolution.BSRGAN.utils import utils_image as util
# from utils import utils_model
from Super_Resolution.BSRGAN.models.network_rrdbnet import RRDBNet as net
def test_bsrgan(model, img_lq, device):
  window_size = 8
  #imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
  #img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
  img_lq = torch.from_numpy(np.transpose(img_lq, (0,3,1,2))/255.).float().to(device)  # CHW-RGB to NCHW-RGB
  #print(img_lq)
  # inference
  print(img_lq.shape)
  with torch.no_grad():
      # pad input image to be a multiple of window_size
      _, _, h_old, w_old = img_lq.size()
      output = model(img_lq)
      img_lq = img_lq.to(device)
      output = output.data.float().cpu().clamp_(0, 1).numpy()
      if output.ndim == 3:
          output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
      output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
      #print(output.shape)
      output = np.transpose(output, (0,2,3,1))
      return output
def define_model_bsrgan(model_path, device, model_name = 'BSRGAN'):
  #model = define_model(args)
  #model_path = os.path.join("Super_Resolution","BSRGAN",'model_zoo', model_name+'.pth')
  model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
  model.load_state_dict(torch.load(model_path), strict=True)
  for k, v in model.named_parameters():
      v.requires_grad = False
  model.eval()
  model = model.to(device)
  return model