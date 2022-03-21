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
import numpy as np

sys.path.insert(0, './ObjDet')
sys.path.insert(0, './age_gender_prediction/regression')
sys.path.append('./age_gender_prediction/VisualizingNDF/regression')
import age_gender_prediction.VisualizingNDF.regression.ndf as ndf

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
from deepface import DeepFace

from age_gender_prediction.VGGFACE.vgg16_model import gender_model

from age_gender_prediction.VisualizingNDF.regression.model import prepare_model
from Super_Resolution.SwinIR import test_swin

from Super_Resolution.BSRGAN import bsrgan_test

parser = argparse.ArgumentParser()

parser.add_argument('--run_on_image', type=bool, default = False, help='If True image else video')
parser.add_argument('--save_csv_location', type=str, default="name_csv_file.csv", help='enter file path of output csv file')
parser.add_argument('--weights_yolo', nargs='+', type=str, default='weights/face_detect/yolov5n-0.5.pt', help='weights path for face detection')
parser.add_argument('--video_image', type=str, default='test.mp4', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--weights_gan', type=str, default ='weights/gan_weights/weights.h5', help='model.pt path(s) for edsr')
parser.add_argument('--output_folder', type=str, default ='output', help='path to save output image/frames')
parser.add_argument('--facelib', type=bool, default =True, help='If true facelib is used else deepface for age, gender prediction')
parser.add_argument('--sr_type', type=int, default =1, help='1->edsr it is the baseline srgan ued, 2->swinir, 3->bcrgan')
parser.add_argument('--deblur_weights', type=str, default ="./", help='Pretrained weights path for debluring models here we used MRNet')
parser.add_argument('--gender_pred', type=int, default = 0, help='If 0 then vgg_face')
parser.add_argument('--gender_weights', type=str, default ="./weights/gender_prediction/gender_model_weights.h5", help='Pretrained weights path for gender prediction')
parser.add_argument('--age_pred', type=int, default = 0, help='If 0 then ndf ')
parser.add_argument('--age_weights', type=str, default ="./weights/age_prediction/age_model_weights.pth", help='Pretrained weights path for gender prediction')
parser.add_argument('--cuda', type=bool, default=False, help='True if want to use cuda')

output_dict = {"frame num":[], "person id":[], "bb_xmin":[], "bb_ymin":[], "bb_height":[], "bb_width":[], "age_min":[], "age_max":[], "age_actual":[], "gender":[] }


opt = parser.parse_args()
# opt.sr_type = 3
# print(opt.sr_type)

# time1 = time.time()
if opt.cuda:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
  device = torch.device("cpu")

model = detect_face.load_model(opt.weights_yolo, device)
#detect_one(model, opt.image, device)


# Load corresponding model architecture and weights
load_file = run_path(os.path.join("Deblur","MPRNet", "Deblurring", "MPRNet.py"))
model_deblur = load_file['MPRNet']()
model_deblur.to(device)

weights = os.path.join("Deblur","MPRNet", "Deblurring", "pretrained_models", "model_"+"deblurring"+".pth")
load_checkpoint(model_deblur, weights, device)
model_deblur.eval()

#######################


age_gender_detector = None
################################

gender_model = gender_model(opt.gender_weights)

########################

############################
class OPT:
  def __init__(self):
    #print(age)
    self.model_type = 'hybrid'
    self.num_output = 128
    self.gray_scale = False
    self.image_size = 256
    self.pretrained = True
    self.n_tree = 5
    self.tree_depth = 6
    if torch.cuda.is_available():
      self.cuda = True 
    else:
      self.cuda = False 
pars = OPT()
#print("age")
age_model = prepare_model(pars)
age_model=torch.load(opt.age_weights)
age_model = age_model.to(device).eval()
##############################

#####load gan weights###########

if opt.sr_type==1:
  model_gan = edsr(scale=4, num_res_blocks=16)
  model_gan.load_weights(opt.weights_gan)
  #print("loaded")
  # model_gan.to(device)


elif opt.sr_type ==2:
  model_gan = test_swin.define_model(opt.weights_gan, device, large_model=True)

elif opt.sr_type ==3:
  model_gan = bsrgan_test.define_model_bsrgan(opt.weights_gan, device)
#################################


import cv2
import time
time1 = time.time()
if not opt.run_on_image:
    
    # Opens the inbuilt camera of laptop to capture video.
    cap = cv2.VideoCapture(opt.video_image)
    i = 0
    p = 0
    s = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))

        
        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break
        
        
        # Save Frame by Frame into disk using imwrite method
        #cv2.imwrite("/content/drive/MyDrive/INTER_IIT_DRIVE/yolov5-face/test_frames/" + 'Frame'+str(i)+'.jpg',frame)
        output_dict = detect_face.detect_one(model, frame, device, depth=16, scale=4, model_gan_path=opt.weights_gan, output_folder=opt.output_folder, frame_num=i, age_gender = opt.facelib, sr = opt.sr_type, model_deblur=model_deblur, output_dict=output_dict, model_gan=model_gan, age_gender_detector=age_gender_detector, gender_model = gender_model, gender_pred=opt.gender_pred, age_model = age_model, age_pred=opt.age_pred)
        i += 1
        #print(i)
        #cv2.imshow('frames',frame)
        #cv2.waitKey(1)
        # if i==5 :
        #   break
        #break

else:
    frame = cv2.imread(opt.video_image)
    output_dict = detect_face.detect_one(model, frame, device, depth=16, scale=4, model_gan_path=opt.weights_gan, output_folder=opt.output_folder, frame_num=0, age_gender = opt.facelib, sr = opt.sr_type, model_deblur=model_deblur, output_dict=output_dict, model_gan=model_gan, age_gender_detector=age_gender_detector, gender_model = gender_model, gender_pred=opt.gender_pred, age_model = age_model, age_pred=opt.age_pred)



time2 = time.time()
print(time2-time1)
df = pd.DataFrame(output_dict)
df.to_csv(opt.save_csv_location) 
#cap.release()
#cv2.destroyAllWindows()

# detect_face.detect_one(model, opt.image, device, depth=16, scale=4, model_gan_path=opt.weights_gan, output_folder=opt.output_folder)


