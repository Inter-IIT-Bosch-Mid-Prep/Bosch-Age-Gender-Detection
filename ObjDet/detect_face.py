import argparse
from selectors import EpollSelector
import time
from pathlib import Path
import os.path
from os import path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
import torchvision.transforms.functional as TF

from deepface import DeepFace

from ObjDet.centroidtracker import CentroidTracker 

from ObjDet.models.experimental import attempt_load
from ObjDet.utils.datasets import letterbox
from ObjDet.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from ObjDet.utils.plots import plot_one_box
from ObjDet.utils.torch_utils import select_device, load_classifier, time_synchronized

from Super_Resolution.sr_gan.model.edsr import edsr
from Super_Resolution.sr_gan.model import resolve_single

from Super_Resolution.bicubic_pytorch import core

from age_gender_prediction import age_gender_pred_deepface

from facelib import FaceDetector, AgeGenderEstimator
from age_gender_prediction import age_gender_pred_facelib

from Deblur.MPRNet.deblurring import restore
from Denoising.Restormer.Motion_Deblurring.mot_deblur import mot_deblur



from Super_Resolution.ESPCN_pytorch.models import ESPCN
from Super_Resolution.ESPCN_pytorch import test

from Super_Resolution.SwinIR.test_swin import test_swinir

from Super_Resolution.BSRGAN.bsrgan_test import test_bsrgan

from Denoising.Restormer.basicsr.models.archs.restormer_arch import Restormer

from age_gender_prediction.VGGFACE.pred_gender import gender_prediction

#os.chdir("./age_gender_prediction/AGE_ENSEMBLE")
#from age_ensembling import *
#os.chdir("./../..")
# age_gender_detector = AgeGenderEstimator()

centd = CentroidTracker(buffer_size=20)
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh_c, conf, landmarks, class_num, j, img_path, sr_model, age_gender, sr, frame_num, output_dict, age_gender_detector, device, gender_model, gender_pred, age_model, age_pred, model_deblur):
    
    # sr = 1
    img_conc = []
    num = 0
    
    for xywh in xywh_c:

        h,w,c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        img_we = (img_path[y1:y2,x1:x2])
        img_we = cv2.resize(img_we, (56, 56))
        img_we = restore(img_we, model_deblur, device)
        num = num+1
        img_conc.append(img_we)
    img_conc = np.array(img_conc)


    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]


    tf = max(tl - 1, 1)  # font thickness
    #label = str(conf)[:5]
    sr_img = img_conc



    if sr==1:  

      sr_img = sr_img[...,::-1]
      sr_img = sr_model(sr_img/1.) 
      sr_img = np.array((sr_img.cpu())).astype("uint8")
      sr_img = sr_img[...,::-1]
      sr_img = np.clip(sr_img, 0, 255)

    elif sr==2:
      sr_img = test_swinir(sr_model, sr_img, device)


    elif sr==3:
      sr_img = test_bsrgan(sr_model, sr_img, device)

    
    gender  = gender_prediction(sr_img, gender_model, device)
    srimg = torch.from_numpy(sr_img.transpose(0, 3, 1, 2))/255.
    age  = age_model(srimg.to(device))[0].data.cpu().detach().numpy()*100

    num = 0
    box=[]
    for xywh in xywh_c:

      h,w,c = img.shape
      tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
      x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
      y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
      x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
      y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)      
      if gender[num] == 0:
        gender_label = "M"
      else:
        gender_label = "W"
      label = "age: " + str(age[num][0]) + ", gender:" + gender_label
      cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
      cv2.putText(img, label, (x1, y1 - 2), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
      output_dict["frame num"].append(frame_num)
      output_dict["person id"].append(num)
      output_dict["bb_xmin"].append(x1)
      output_dict["bb_ymin"].append(y2)
      output_dict["bb_height"].append(y2-y1)
      output_dict["bb_width"].append(x2-x1)
      output_dict["age_min"].append((age[num][0]//10) * 10)
      output_dict["age_max"].append(((age[num][0]//10) * 10)+10)
      output_dict["age_actual"].append(age[num][0])
      if gender[num]==0:
       output_dict["gender"].append("M")
      else:
       output_dict["gender"].append("F")
      box.append([x1,y2,y2-y1,x2-x1])
      num = num+1
    
    iddet = centd.Update(box,SEPARATION_DISTIANCE=40.0,REUSE_IDS=False)
    if iddet is not None:
      for val in iddet:
        cv2.circle(img,(val[0],val[1]),5,[0,0,255],-1)
        cv2.putText(img, str(val[3]), (val[0]-5,val[1]-15), cv2.FONT_HERSHEY_SIMPLEX , 1, [0,0,255], 4, cv2.LINE_AA)

	


    return img, output_dict



def detect_one(model, image_path, device, depth, scale, model_gan_path, output_folder, frame_num, age_gender, sr, model_deblur, output_dict, model_gan, age_gender_detector, gender_model, gender_pred, age_model, age_pred):


    img_size = max(image_path.shape[0], image_path.shape[1])
    conf_thres = 0.3
    iou_thres = 0.5
    
    img_path_copy = np.copy(image_path)
    orgimg = image_path


    orgimg = image_path
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        print("bush")
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    #print(imgsz.shape)
    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
            xywh = []
            for j in range(det.size()[0]):
                xywh.append((xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist())
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()

            if(len(xywh)>0):
              print(len(xywh))
              orgimg, output_dict = show_results(orgimg, xywh, conf, landmarks, class_num, j, img_path_copy, sr_model = model_gan, age_gender=age_gender, sr=sr, output_dict=output_dict, frame_num= frame_num, age_gender_detector=age_gender_detector, device=device, gender_model = gender_model, gender_pred=gender_pred, age_model=age_model, age_pred=age_pred, model_deblur = model_deblur)
    
    output_path = output_folder + '/frame_' + str(frame_num) + '.jpg'
    print(output_path)
    cv2.imwrite( output_path , orgimg)
    return output_dict