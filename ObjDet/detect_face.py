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
import numpy as np
import torchvision.transforms.functional as TF

from deepface import DeepFace

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

# age_gender_detector = AgeGenderEstimator()

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

def show_results(img, xywh_c, conf, landmarks, class_num, j, img_path, sr_model, age_gender, sr, frame_num, output_dict, age_gender_detector, device, gender_model, gender_pred, age_model, age_pred):
    
    # print("shanmukh")
    # print(sr)
    # print("shanmukh")

    #print(img.shape)
    #age_gender_detector = AgeGenderEstimator()
    # if path.exists(img_path[:-5]) == False:
    #      os.mkdir(img_path[:-5])
    img_conc = []
    for xywh in xywh_c:

        h,w,c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        img_we = (img_path[y1:y2,x1:x2])
        img_we = cv2.resize(img_we, (56, 56))
        img_conc.append(img_we)
    img_conc = np.array(img_conc)
    #print("rrrrrrrrrrrrrrrrrrrrrr")
    #print(img_conc.shape)
    # cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    # print("*****")
    # print(x1-x2)
    # print(y1-y2)
    # print("******")
    # for i in range(5):
    #     point_x = int(landmarks[2 * i] * w)
    #     point_y = int(landmarks[2 * i + 1] * h)
    #     cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    #label = str(conf)[:5]
    sr_img = img_conc
    # print("************")
    # print(x2-x1)
    # print(y2-y1)
    # print("************")
    if sr==0:
      #print("True")
      #print(sr_img.shape)
      #print(core.imresize(TF.to_tensor(sr_img).unsqueeze_(0), scale=4)[0])
      sr_img = np.array(255*(core.imresize(TF.to_tensor(sr_img).unsqueeze_(0), scale=4)[0].permute(1,2,0))).astype('uint8')
      cv2.imwrite("/content/sample_data/sr0.jpg", sr_img)

    elif sr==1:  
      c = time.time()
      sr_img = sr_model(sr_img/255.)
      #np.squeeze(sr_img)
      #print(sr_img)
      # path = "/content/super_res/" + "frame" +"_"+ str(frame_num) + "_" + str(j) + ".jpg"
      # cv2.imwrite(path, np.array((sr_img[0].cpu())))
      sr_img = np.array((sr_img.cpu()*255)).astype("uint8")
      sr_img = np.clip(sr_img, 0, 255)
      #print(sr_img.shape)
      #num = 0
      for num in range(sr_img.shape[0]):
        path = "/content/face_cuts_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(num) + ".jpg"
        cv2.imwrite(path, np.array(sr_img[num]))
      d = time.time()
      print("s_r : ", d-c)

      #print(sr_img)
      # path = "/content/face_cuts_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(j) +"_"+str(age)+"_"+ str(gender)+ ".jpg"
      # cv2.imwrite(path, np.array(sr_img))

    elif sr==2:
      sr_img = test.espch_sr(sr_img, sr_model, scale=3)

    elif sr==3:
      #print("cacke")
      #print(sr_img.shape)
      c = time.time()
      sr_img = test_swinir(sr_model, sr_img, device)
      d = time.time()
      for num in range(sr_img.shape[0]):
        path = "/content/face_cuts_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(num) + ".jpg"
        cv2.imwrite(path, np.array(sr_img[num]))
      print("s_r : ", d-c)
      #print(sr_img.shape)
      # path = "/content/deblur_swinir/" + "frame" +"_"+ str(frame_num) + ".jpg"
      # cv2.imwrite(path, sr_img)

    elif sr==4:
      #p
      #print("INTERIIT")
      sr_img = test_bsrgan(sr_model, sr_img, device)
      #print(sr_img.shape)
      # path = "/content/deblur_swinir/" + "frame" +"_"+ str(frame_num) + ".jpg"
      # cv2.imwrite(path, sr_img)

      #print(sr_img)
      #cv2.imwrite("/content/sample_data/sr2.jpg", sr_img)


  
    
    #print(sr_img)
    # sr_img = DeepFace.detectFace(sr_img, target_size = (224, 224), detector_backend = 'opencv', enforce_detection = False, align = True)
    # print(sr_img)
    # path = "/content/face_alignment/" + "frame" +"_"+ str(frame_num) + "_" + str(j) + ".jpg"
    # cv2.imwrite(path, sr_img)  
    # path = "/content/face_cuts_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(j) +"_"+str(age)+"_"+ str(gender)+ ".jpg"
    # cv2.imwrite(path, np.array(sr_img))   
    
    # if age_gender==False:
    #   gender , age , agebucket = age_gender_pred_deepface.calculate_gender(sr_img)
    # # print("hey")
    # # path = "/content/face_cuts_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(j) +"_"+str(age)+"_"+ str(gender)+ ".jpg"
    # # cv2.imwrite(path, np.array(sr_img))

    # else :
    #   #print("Shanmukh")
    #   gender , age , agebucket = age_gender_pred_facelib.calculate_gender(sr_img, age_gender_detector, device)
    
    c = time.time()
    if gender_pred==0:
      gender  = gender_prediction(sr_img, gender_model, device)
      print(gender.shape)
    d = time.time()
    print("gender : ", d-c)
    
    #print(sr_img)
    c = time.time()
    if age_pred==0:
      # srimg = cv2.resize(sr_img, (200,200))
      # srimg = torch.Tensor(srimg)
      srimg = torch.from_numpy(sr_img.transpose(0, 3, 1, 2))/255.
      # srimg = torch.reshape(srimg, (1,3,200,200))
      #print(age)
      print(srimg.shape)
      #print(len(age_model(srimg.to(device))))
      #print(age)
      age  = age_model(srimg.to(device))[0].data.cpu().detach().numpy()*100
      print(age.shape)
      
    d = time.time()
    print("age : ", d-c)
    #print("ttttttttttttt")
    #print(sr_img)
    #copy_img = np.copy(img)
    
    num = 0
    for xywh in xywh_c:

      h,w,c = img.shape
      tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
      x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
      y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
      x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
      y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)      
      label = "age: " + str(age[num][0]) + ", gender:" + str(gender[num])
      cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
      cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
      num = num+1
      cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
      cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
      #cv2.rectangle(copy_img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
      label = "age: " + str(age) + ", gender:" + str(gender)
      output_dict["frame num"].append(frame_num)
      output_dict["person id"].append(num)
      output_dict["bb_xmin"].append(x1)
      output_dict["bb_ymin"].append(y2)
      output_dict["bb_height"].append(y2-y1)
      output_dict["bb_width"].append(x2-x1)
      output_dict["age_min"].append((age//10) * 10)
      output_dict["age_max"].append(((age//10) * 10)+10)
      output_dict["age_actual"].append(age)
      output_dict["gender"].append(gender)
    #print(sr_img.shape)
    # path = "/content/face_cuts_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(j) +"_"+str(age)+"_"+ str(gender)+ ".jpg"
    # cv2.imwrite(path, np.array(sr_img))

    # path = "/content/face_cuts_wo_sr/" + "frame" +"_"+ str(frame_num) + "_" + str(j) +"_"+str(age)+"_"+ str(gender)+ ".jpg"
    # cv2.imwrite(path, img_path[y1:y2,x1:x2])
    # # print("*************************")
    # # print(y2-y1)
    # # print(x2-x1)
    # # print("*************************")
    # cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
    # cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    #print(img_path[:-5] + '/cut'+ str(j) +'.jpg')
    #cv2.imwrite('/content/output/'+ img_path[-13:-4]  + '/cut'+ str(j) +'.jpg', img[y1:y2,x1:x2])
    return img, output_dict



def detect_one(model, image_path, device, depth, scale, model_gan_path, output_folder, frame_num, age_gender, sr, model_deblur, deblur_type, output_dict, model_gan, age_gender_detector, gender_model, gender_pred, age_model, age_pred):
    # Load model
    #print(deblur_type)
    #model_deblur = model_deblur
    img_size = max(image_path.shape[0], image_path.shape[1])
    conf_thres = 0.3
    iou_thres = 0.5
    
    # print("@@@@@@@@@@@@@@@@@@@@")
    # print(image_path.shape)
    # print("@@@@@@@@@@@@@@@@@@@@")
    
    # if sr==0:
    #   #print("Shanmukh")
    #   model_gan = None

    # elif sr==1:
    #   model_gan = edsr(scale=scale, num_res_blocks=depth)
    #   model_gan.load_weights(model_gan_path)

    # elif sr==2:
    #   model_gan = ESPCN(scale_factor = 3).to(device)
    #   state_dict = model_gan.state_dict()
    #   print(model_gan_path)
    #   for n, p in torch.load(model_gan_path, map_location=lambda storage, loc: storage).items():
    #       if n in state_dict.keys():
    #           state_dict[n].copy_(p)
    #       else:
    #           raise KeyError(n)

    img_path_copy = np.copy(image_path)
    orgimg = image_path
    #orgimg = cv2.imread(image_path)  # BGR
    if deblur_type==0:
      c = time.time()
      orgimg = restore(image_path, model_deblur, device)
      d = time.time()
      print("deblur : ", d-c)

      path = "/content/deblur/" + "frame" +"_"+ str(frame_num) + ".jpg"
      #print("tyghdsvdsfugfvsukfysg")
      cv2.imwrite(path, orgimg)

    
    elif deblur_type==1:
      #orgimg = mot_deblur(image_path, model_deblur)
      pass


    # c = time.time()
    # orgimg = cv2.fastNlMeansDenoisingColored(orgimg, None, 10, 10, 7, 15)
    # d = time.time()
    # print("denoise : ", d-c)
    # path = "/content/denoise_opencv/" + "frame" +"_"+ str(frame_num) + ".jpg"
    # cv2.imwrite(path, orgimg)   

    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sr_img = cv2.filter2D(sr_img, -1, kernel)

    # path = "/content/sharpen_opencv/" + "frame" +"_"+ str(frame_num) + ".jpg"
    # cv2.imwrite(path, sr_img)  

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
    c = time.time()
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    d = time.time()
    print("yolov5 : ", d-c)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        print(len(det))
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
                #print(j)
                path = "/content/out/" + "frame" +"_"+ str(frame_num)+"_"+str(j) + ".jpg"
                cv2.imwrite(path, orgimg)
                path = "/content/out_check/" + "frame" +"_"+ str(frame_num)+"_"+str(j) + ".jpg"
                cv2.imwrite(path, img_path_copy)
            orgimg, output_dict = show_results(orgimg, xywh, conf, landmarks, class_num, j, img_path_copy, sr_model = model_gan, age_gender=age_gender, sr=sr, output_dict=output_dict, frame_num= frame_num, age_gender_detector=age_gender_detector, device=device, gender_model = gender_model, gender_pred=gender_pred, age_model=age_model, age_pred=age_pred)
    
    output_path = output_folder + '/frame_' + str(frame_num) + '.jpg'
    print(output_path)
    cv2.imwrite( output_path , orgimg)
    return output_dict