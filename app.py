import streamlit as st 
import cv2
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
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
from PIL import Image
st.set_page_config(layout="wide")
sys.path.append('./age_gender_prediction.VisualizingNDF/regression')
from age_gender_prediction.VisualizingNDF.regression import *
import ndf

# import age_gender_prediction.VisualizingNDF.regression.ndf as ndf
# print(sys.path)
import sys
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
from deepface import DeepFace

from age_gender_prediction.VGGFACE.vgg16_model import gender_model as gen_model

from age_gender_prediction.VisualizingNDF.regression.model import prepare_model
from Super_Resolution.SwinIR import test_swin

from Super_Resolution.BSRGAN import bsrgan_test



def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():

    st.markdown("<h1 style='text-align: center; font-size:80px;'>WELCOME TO THE FOURIER WORLD</h1>", unsafe_allow_html=True)
    html_temp = """
            <br>
            <div>
            <img src='https://user-images.githubusercontent.com/60772854/159160166-49a446da-922f-4e27-b3a8-696ec6ed67ed.gif'   style="width:100%;height: 810px;">
            </div>
           
            """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size:80px;'>Upload a Image for Testing</h1>", unsafe_allow_html=True)
    
    # page_bg_img = '''
    #         <style>
    #         body {
    #         background-image: url("https://user-images.githubusercontent.com/60772854/159160166-49a446da-922f-4e27-b3a8-696ec6ed67ed.gif");
    #         background-size: cover;
    #         }
    #         </style>
    #         '''

    # st.markdown(page_bg_img, unsafe_allow_html=True)
    output_dict = {"frame num":[], "person id":[], "bb_xmin":[], "bb_ymin":[], "bb_height":[], "bb_width":[], "age_min":[], "age_max":[], "age_actual":[], "gender":[] }
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    gans_list = ['Bi-Cubic Interpolation','EDSR','ESPCN','SwinIR','BSRGAN']
    sr_gan = st.selectbox('Select Super Resolution Methods',gans_list,index=1)

    if image_file is not None:
      my_bar = st.progress(0)
      # To See details
      file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
      #st.write(file_details)
      col1, col2= st.columns(2)
            # To View Uploaded Image
      with col1:
        st.success("Input Image")
        st.image(load_image(image_file),use_column_width=True)


      #opt = {}
      class PARS:
        def __init__(self):
            self.run_on_image = True
            self.weights = "./weights/face_detect/yolov5n-0.5.pt"
            self.video_image =  image_file
            # self.img-size = 640
            self.weights_gan = [
                                "./weights/gan_weights/weights.h5"
                                ]
            #self.weights_espcn = 
            self.output_folder = "output_frames"
            self.facelib = True
            self.bic_inter = 0
            self.deblur_weights = "./weights/deblur/model_deblurring.pth"
            self.deblur_type = 0
            self.gender_pred = 0
            self.gender_weights = "./weights/gender_prediction/gender_model_weights.h5"
            self.age_pred = 0
            self.age_weights = "./weights/age_prediction/age_model_weights.pth"
      opt = PARS()
  
      device = torch.device("cpu")
      model = detect_face.load_model(opt.weights, device)
        # detect_one(model, opt.image, device)
      my_bar.progress(20)

      # Load corresponding model architecture and weights
      if opt.deblur_type==0:
          load_file = run_path(os.path.join("Deblur","MPRNet", "Deblurring", "MPRNet.py"))
          model_deblur = load_file['MPRNet']()
          model_deblur.to(device)

          weights = os.path.join("Deblur","MPRNet", "Deblurring", "pretrained_models", "model_"+"deblurring"+".pth")
          load_checkpoint(model_deblur, weights, device)
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
          # print(opt.deblur_weights_restormer, "))))))))))))))))")
          checkpoint = torch.load(opt.deblur_weights_restormer)
          model_deblur.load_state_dict(checkpoint['params'])
          #print("===>Testing using weights: ",opt.deblur_weights_restormer)
          model_deblur.to(device)
          #model_restoration = nn.DataParallel(model_restoration)
          model_deblur.eval()

      else :
        model_deblur = None
      #   #######################

      ############################
      age_gender_detector = None #AgeGenderEstimator()
      ############################

      ########################

      # gend_model = None
      if opt.gender_pred==0:
        #print("srgdfh")
        gender_model = gen_model(opt.gender_weights)

      ########################

      ############################
      gend_model = None
      if opt.age_pred==0:
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
            self.cuda = False 
        pars = OPT()
        #print("age")
        age_model = prepare_model(pars)
        print(opt.age_weights)
        age_model=torch.load(opt.age_weights)
        age_model = age_model.to(device).eval()
      ##############################

      #####load gan weights###########
      if opt.bic_inter==0:
        model_gan = edsr(scale=4, num_res_blocks=16)
        model_gan.load_weights(opt.weights_gan[opt.bic_inter])
        #print("loaded")
        # model_gan.to(device)

      # elif opt.bic_inter ==3:
      #   #print("HEY GHYBHDJIN")
      #   model_gan = test_swin.define_model(opt.weights_gan[opt.bic_inter], device, large_model=True)

      # elif opt.bic_inter ==4:
      #   #print("HEY SHANMUKH")
      #   model_gan = bsrgan_test.define_model_bsrgan(opt.weights_gan[opt.bic_inter], device)
      #################################
      my_bar.progress(40)

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
              frame = cv2.resize(frame, (800, 800))
              frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
              print(frame.shape)
              
              # This condition prevents from infinite looping
              # incase video ends.
              if ret == False:
                  break
              
              path = "/content/initial/" + "frame" +"_"+ str(i) + ".jpg"
              cv2.imwrite(path, frame)
              
              # Save Frame by Frame into disk using imwrite method
              #cv2.imwrite("/content/drive/MyDrive/INTER_IIT_DRIVE/yolov5-face/test_frames/" + 'Frame'+str(i)+'.jpg',frame)
              output_dict = detect_face.detect_one(model, frame, device, depth=16, scale=4, model_gan_path=opt.weights_gan[opt.bic_inter], output_folder=opt.output_folder, frame_num=i, age_gender = opt.facelib, sr = 1, model_deblur=model_deblur, output_dict=output_dict, model_gan=model_gan, age_gender_detector=age_gender_detector, gender_model = gender_model, gender_pred=opt.gender_pred, age_model = age_model, age_pred=opt.age_pred)
              i += 1
              #print(i)
              #cv2.imshow('frames',frame)
              #cv2.waitKey(1)
              # if i==5 :
              #   break
              #break

      else:
          img = load_image(image_file)
          frame = np.array(img)
          output_dict = detect_face.detect_one(model, frame, device, depth=16, scale=4, model_gan_path=opt.weights_gan[opt.bic_inter], output_folder=opt.output_folder, frame_num=0, age_gender = opt.facelib, sr = 1, model_deblur=model_deblur, output_dict=output_dict, model_gan=model_gan, age_gender_detector=age_gender_detector, gender_model = gender_model, gender_pred=opt.gender_pred, age_model = age_model, age_pred=opt.age_pred)



          time2 = time.time()
          print(time2-time1)
          df = pd.DataFrame(output_dict)
          print(output_dict)
          my_bar.progress(90)
          for i in range(len(output_dict['person id'])):
            start_point=(output_dict['bb_xmin'][i],output_dict['bb_ymin'][i])
            end_point=(output_dict['bb_xmin'][i]+output_dict['bb_width'][i],output_dict['bb_ymin'][i]-+output_dict['bb_height'][i])
            color = (255, 0, 0)
            thickness = 2
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
          my_bar.progress(100)
          with col2:
            st.success("Output Image")
            st.image(frame,use_column_width=True)
          df.to_csv('name_csv_file.csv') 
    
    

if __name__=='__main__':
    main()
    
    
    