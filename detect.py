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
parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--weights_gan', type=str, default ='runs/train/exp5/weights/last.pt', help='model.pt path(s) for gan')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detect_face.load_model(opt.weights, device)
#detect_one(model, opt.image, device)

detect_face.detect_one(model, opt.image, device, depth=16, scale=4, model_gan_path=opt.weights_gan)


