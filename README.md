# BOSCH’S AGE AND GENDER DETECTION
## Overview
    Over the recent years, detecting human beings in a video scene of a surveillance system is 
    attracting more attention due to its wide range of applications in abnormal event detection, 
    human gait characterization, person counting in a dense crowd, person identification, 
    gender classification, fall detection for elderly people, etc.

## Description 
    The scenes obtained from a surveillance video are usually with low resolution. 
    Most of the scenes captured by a static camera are with minimal change of background. 
    Objects in outdoor surveillance are often detected in far-fields. Most existing digital 
    video surveillance systems rely on human observers for detecting specific activities in a 
    real-time video scene. However, there are limitations in the human capability to monitor 
    simultaneous events in surveillance displays. Hence, human motion analysis in automated video 
    surveillance has become one of the most active and attractive research topics in the 
    area of computer vision and pattern recognition.

## Problem Statement
    Build a solution to estimate the gender and age of people from a surveillance video 
    feed (like mall, retail store, hospital etc.). Consider low resolution cameras as well as 
    cameras put at a height for surveillance.


# How To Run
    python --weights <PATH_TO_WEIGHTS_of_YOLO_V%5 --video <PATH_TO_VIDEO> --img-size <INFERENCE_SIZE_IN_PIXELS> --weight_gan <PATH_TO_WEIGHTS_OF_GAN> --output_folder <PATH_TO_SAVE_OUTPUT_IMAGES>

# Venv To Run The Code
    $ conda create --name age_and_gender_detection python=3.8
    $ conda activate age_and_gender_detection
    $ pip install opencv-python
    $ conda install pytorch torchvision torchaudio cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch
    $ conda install -c conda-forge tqdm
    $ pip install pyyaml
    $ conda install -c conda-forge matplotlib
    $ conda install -c anaconda pandas
    $ conda install -c anaconda seaborn
    $ pip install tensorflow
    $ pip install deepface

# Where To Look For What

