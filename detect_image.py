import argparse
import time
from pathlib import Path
import os.path
from os import path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import sys
from runpy import run_path
import pandas as pd

sys.path.insert(0, './ObjDet')

from ObjDet.models.experimental import attempt_load
from ObjDet.utils.datasets import letterbox
from ObjDet.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from ObjDet.utils.plots import plot_one_box
from ObjDet.utils.torch_utils import select_device, load_classifier, time_synchronized
from ObjDet import detect_face

from Deblur.MPRNet.deblurring import load_checkpoint

from Denoising.Restormer.basicsr.models.archs.restormer_arch import Restormer

from Super_Resolution.ESPCN_pytorch.models import ESPCN
from Super_Resolution.ESPCN_pytorch import test

from Super_Resolution.sr_gan.model.edsr import edsr
from Super_Resolution.sr_gan.model.common import resolve_single

from Super_Resolution.bicubic_pytorch import core
from facelib import FaceDetector, AgeGenderEstimator
from age_gender_prediction import age_gender_pred_facelib



parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
parser.add_argument('--image', type=str, default='data/images/test.avi', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--weights_gan', type=str, default ='runs/train/exp5/weights/last.pt', help='model.pt path(s) for edsr')
#parser.add_argument('--weights_espcn', type=str, default ='runs/train/exp5/weights/last.pt', help='model.pt path(s) for espcn')
parser.add_argument('--output_folder', type=str, default ='runs/train/output/', help='path to save output image')
parser.add_argument('--facelib', type=bool, default =True, help='If true facelib is used else deepface for age, gender prediction')
parser.add_argument('--bic_inter', type=int, default =0, help='If true bicubic is used else esdr for sr')
parser.add_argument('--deblur_weights', type=str, default ="./", help='Pretrained weights path for debluring models')
parser.add_argument('--deblur_weights_restormer', type=str, default ="/content/drive/MyDrive/Bosch-Age-Gender-Detection-object-detection/Bosch-Age-Gender-Detection-object-detectionanddenoising/Denoising/Restormer/Motion_Deblurring/motion_deblurring.pth", help='Pretrained weights path for debluring models')
parser.add_argument('--deblur_type', type=int, default = 2, help='If 1 then restormer')

output_dict = {"frame num":[], "person id":[], "bb_xmin":[], "bb_ymin":[], "bb_height":[], "bb_width":[], "age_min":[], "age_max":[], "age_actual":[], "gender":[] }


opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detect_face.load_model(opt.weights, device)
#detect_one(model, opt.image, device)


# Load corresponding model architecture and weights
if opt.deblur_type==0:
    load_file = run_path(os.path.join("Deblur","MPRNet", "Deblurring", "MPRNet.py"))
    model_deblur = load_file['MPRNet']()
    model_deblur.to(device)

    weights = os.path.join("Deblur","MPRNet", "Deblurring", "pretrained_models", "model_"+"deblurring"+".pth")
    load_checkpoint(model_deblur, weights)
    model_deblur.eval()

#motion deblur
elif opt.deblur_type==1:
    ####### Load yaml #######
    yaml_file = 'Denoising/Restormer/Motion_Deblurring/Options/Deblurring_Restormer.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################

    model_deblur = Restormer(**x['network_g'])

    checkpoint = torch.load(opt.deblur_weights_restormer)
    model_deblur.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ",opt.deblur_weights_restormer)
    model_deblur.to(device)
    #model_restoration = nn.DataParallel(model_restoration)
    model_deblur.eval()

else :
  model_deblur = None
#######################

############################
age_gender_detector = AgeGenderEstimator()
############################

#####load gan weights###########
if opt.bic_inter==0:
  #print("Shanmukh")
  model_gan = None

elif opt.bic_inter==1:
  model_gan = edsr(scale=4, num_res_blocks=16)
  model_gan.load_weights(opt.weights_gan)
  print("loaded")
  # model_gan.to(device)

elif opt.bic_inter==2:
  model_gan = ESPCN(scale_factor = 3).to(device)
  state_dict = model_gan.state_dict()
  print(opt.weights_gan)
  for n, p in torch.load(opt.weights_gan, map_location=lambda storage, loc: storage).items():
      if n in state_dict.keys():
          state_dict[n].copy_(p)
      else:
          raise KeyError(n)
#################################


import cv2
import time
 
# Opens the inbuilt camera of laptop to capture video.
#cap = cv2.VideoCapture(opt.video)
i = 0
p = 0
s = time.time()
# while(cap.isOpened()):
#     ret, frame = cap.read()
     
#     # This condition prevents from infinite looping
#     # incase video ends.
#     if ret == False:
#         break
     
    # Save Frame by Frame into disk using imwrite method
    #cv2.imwrite("/content/drive/MyDrive/INTER_IIT_DRIVE/yolov5-face/test_frames/" + 'Frame'+str(i)+'.jpg',frame)
x = cv2.imread(opt.image)
output_dict = detect_face.detect_one(model, x, device, depth=16, scale=4, model_gan_path=opt.weights_gan, output_folder=opt.output_folder, frame_num=i, age_gender = opt.facelib, sr = opt.bic_inter, model_deblur=model_deblur, deblur_type=opt.deblur_type, output_dict=output_dict, model_gan=model_gan, age_gender_detector=age_gender_detector)
    # i += 1
    # #print(i)
    # #cv2.imshow('frames',frame)
    # cv2.waitKey(1)
    # # if i==5 :
    # #   break
    # break

df = pd.DataFrame(output_dict)
df.to_csv('output.csv') 
#cap.release()
cv2.destroyAllWindows()