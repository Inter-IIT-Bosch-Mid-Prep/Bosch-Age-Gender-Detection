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

sys.path.insert(0, './ObjDet')

from ObjDet.models.experimental import attempt_load
from ObjDet.utils.datasets import letterbox
from ObjDet.utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from ObjDet.utils.plots import plot_one_box
from ObjDet.utils.torch_utils import select_device, load_classifier, time_synchronized
from ObjDet import detect_face



parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='path to weights of YOLO_V5')
parser.add_argument('--video', type=str, default='data/images/test.avi', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--weights_gan', type=str, default ='runs/train/exp5/weights/last.pt', help='model.pt path(s) for gan')
parser.add_argument('--output_folder', type=str, default ='runs/train/output/', help='path to save output image')


opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detect_face.load_model(opt.weights, device)
#detect_one(model, opt.image, device)

import cv2
import time
 
# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(opt.video)
i = 0
p = 0
s = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
     
    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
     
    # Save Frame by Frame into disk using imwrite method
    #cv2.imwrite("/content/drive/MyDrive/INTER_IIT_DRIVE/yolov5-face/test_frames/" + 'Frame'+str(i)+'.jpg',frame)
    detect_face.detect_one(model, frame, device, depth=16, scale=4, model_gan_path=opt.weights_gan, output_folder=opt.output_folder, frame_num=i)
    i += 1
    print(i)
    #cv2.imshow('frames',frame)
    cv2.waitKey(1)
    #break
 
cap.release()
cv2.destroyAllWindows()

# detect_face.detect_one(model, opt.image, device, depth=16, scale=4, model_gan_path=opt.weights_gan, output_folder=opt.output_folder)


