# BOSCH’S AGE AND GENDER DETECTION

<!-- ![](imgs/logo.png) -->

## Overview
  The scenes obtained from a surveillance video are usually with low resolution. Most existing digital video surveillance systems rely on human observers for detecting specific activities in a real-time video scene. However, there are limitations in the human capability to monitor simultaneous events in surveillance displays. Low-quality images require super resolution techniques to be visually perceptible and then we can use age and gender estimation techniques for a wide range of applications like abnormal
event detection, person counting in a dense crowd, person identification, gender classification, for elderly people.

## Problem Statement
  Build a solution to estimate the gender and age of people from a surveillance video 
  feed (like mall, retail store, hospital etc.). Consider low resolution cameras as well as 
  cameras put at a height for surveillance.
## Table of Contents
1. [Installation](#installation)
2. [How To Run](#how-to-run)
3. [Eval Datasets](#evaluation-datasets)
4. [Methodology](#methodology)
5. [Results](#results)
6. [File Structure](#file-structure)
# Installation

1. Clone the repository  
    ```bash 
    git clone https://github.com/Inter-IIT-Bosch-Mid-Prep/Bosch-Age-Gender-Detection-IITKGP.git
    ```
2. Run the following command to create the proper conda environment with all the required dependencies  
    ```bash 
    cd Bosch-Age-Gender-Detection-IITKGP 
    conda env create -f env.yml
    ```
3. Activate the conda environment  
    ```bash 
    conda activate bosch
    ```
4. Download the finetuned-weights for UTK NDF for age prediction from [here](https://drive.google.com/file/d/1cL4QU0jXwj60E753_fHJEJAnG8lv7LOD/view?usp=sharing) and extract it to the root folder ```${root}/```.
5. Download the VGGFace Gender weights from [here](https://drive.google.com/file/d/1jbE2RDVM_oPZSs88f1kLP-k9xeGmh0AE/view?usp=sharing) and extract it to the ```${root}```
# How To Run
To run the entire pipeline on a single video you can use the below command.
    ```
    python detect.py --weights <PATH_TO_WEIGHTS_of_YOLO_V5> --video <PATH_TO_VIDEO> --img-size <INFERENCE_SIZE_IN_PIXELS> --weight_gan <PATH_TO_WEIGHTS_OF_GAN> --output_folder <PATH_TO_SAVE_OUTPUT_IMAGES>
    ```   
Note that all cofigurations are optional here. To run the etire pipeline with default configuration on test.mp4, run the following command :-
    ```
    python detect.py 
    ```   

<!-- By default ```detect.py``` will run on test.mp4 -->

# Evaluation Datasets
For testing our pipeline and individual blocks we have come up with some open-source datasets as well as our manually collected datasets.

The opensource datasets can be found below and the manually collected datasets can be found [here](#TODO)

| Task                      | Dataset Link                                                                                                 |
|---------------------------|--------------------------------------------------------------------------------------------------------------|
| Face Detection            | [WiderFace](http://shuoyang1213.me/WIDERFACE/)                                                               |
|                           | [FDDB](http://vis-www.cs.umass.edu/fddb/)                                                                    |
| Age and Gender Estimation | [UTKFace](https://susanqq.github.io/UTKFace/#:~:text=Introduction,%2C%20occlusion%2C%20resolution%2C%20etc.) |
|                           | [CACD](https://bcsiriuschen.github.io/CARC/)                                                                 |
|                           | [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)                              |
|                           | [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)                                            |

We also tested our Face Detection algorithms 

# Methodology 

## Initial Preprocessing
We are initially extracting the individual frames from the given input video and then applying state of the art denoising methods. We have provided an option to the user to apply denoising methods such as [Restormer](https://github.com/swz30/Restormer) or [HINet](https://github.com/megvii-model/HINet).

## Face Detection
We analysed the problem statement from various perspectives and finally decided to go ahead with face detection.

1. Firstly, surveillance videos usually record humans from a height which obfuscates information about height and pose. Applying height and pose estimation algorithms on the top of person detection algorithms would have proven computationlly expensive.
2. Secondly clothing information would have introduced a stereotypical bias which would have been harmful for marginalized groups. 

Thus we decided to use Face detection algorithms. For our pipeline we have integrated [Yolov5-face](https://github.com/deepcam-cn/yolov5-face). We had also tried out various other object detection algorithms however in our own test datasets we did not get satisfactory results in terms of Mean Accuracy Precision(mAP).

| Model      | mAP   |  
|------------|-------|
| TinaFace   | 94.17 |
| YoloV5     | 95.99 |  
| RetinaFace | 91.45 |  
## Super Resolution

### Preprocessing for Super Resolution
We have extensively tested super resolution algorithms and realised that the extracted images usually contained blurred photos which rendered the super resolution algorithms useless and gave sub-par performance on age and gender tasks. Thus we added a new preprocessing block of [Deblurring](https://github.com/swz30/MPRNet)

We have provided comparision for Wall Time, PSNR and SSIM accross multiple Super Resolutiom methods 
| MODEL           | Custom Metric | Wall Time    | PSNR        |
| --------------- | ------------- | ------------ | ----------- |
|                 |               |              |             |
| WDSR            | 345.5003798   | 15.85164261  | 30.36618739 |
| EDSR            | 347.6678516   | 2.112349987  | 30.34467902 |
| SRGAN           | 354.7159776   | 9.196819544  | 29.42405326 |
| FSRCNN          | 430.6859193   | 0.3795514107 | 23.69700551 |
| RDN             | 307.7455076   | 0.3795514107 | 24.58058639 |
| SRDenseNet      | 408.9996247   | 17.12142944  | 24.05288471 |
| ESPCN           | 362.0292181   | 0.4130887985 | 25.14499845 |
| FSRCNN\_trained | 575.1895184   | 0.8021452427 | 21.94459659 |

### Weighted Frequency Aware Super Resolution Algorithm
We have addressed the main pain point of the Problem statement. Just to give an overview, the existing super resolutions algorithms provided a high Peak Signal-to-Noise Ratio(PSNR) value but failed to preserve high frequency details of the image.
Also existing super resolution algorithms are usually modifications of SRGANs which require expensive computation to train and have loss convergence issues. 

```Thus we introduced a novel technique where we introduce a new loss in addition to exisitng reconstruction loss without introducing any new network parameters. Thus we follow the same training procedure but optimize the parameters with respect to the new loss which actually helps in preserving the high frequency components. ```

The loss is formulated as ![](imgs/loss.png)

## Age and Gender Prediction
We have also done extensive experimentation on age and gender prediction. First we did a sanity check whether super resolution was useful for our task hence we ran benchmarks tests with and without super resolution whilst considering VGGFace as the classification model. The results are shown below
| Image size | No SR | BSRGAN | EDSR  | SwinIR |
|------------|-------|--------|-------|--------|
| 7x7        | 0.287 |  0.241 | 0.314 |  0.252 |
| 14X14      | 0.352 |  0.313 | 0.386 |  0.313 |
| 28x28      | 0.488 |  0.499 | 0.523 |  0.495 |
| 56x56      | 0.513 | 0.5342 | 0.551 |  0.533 |

This shows us a general increase in accuracy for age prediction in the case of EDSR accross all the image sizes. We have taken all possible image sizes as Yolov5-face returns faces with different dimensions from the range of 7x7 to 96x96.

- For the gender classification task since we only have 2 labels using a deeper and complex model would overfit to the data hence we train some layers of the original VGGFace model which reported a test accuracy of 94% and we are using that in our final pipeline.  

### Gender
| Model                   | Accuracy |  Recall   | Precision | F1      | False Positive Rate |
| ----------------------- | -------- |  -------- | --------- | ------- | ------------------- |
| Deep Face (Retina Face) | 0.67123  |  0.68     | 0.68811   | 0.66945 | 0.4423              |
| Deep Face (Opencv)      | 0.672523 |  0.68303  | 0.696     | 0.66939 | 0.46332             |
| Deep Face (SSD)         | 0.65992  |  0.67072  | 0.68387   | 0.65633 | 0.47964             |
| InsightFace             | 0.72428  |  0.72254  | 0.72332   | 0.72282 | 0.2487              |
| FaceLib                 | 0.73386  |  0.735218 | 0.73404   | 0.73357 | 0.283954            |

- For the age classification task, we faced quite a number of challenges such as 
  1. Non-uniformity in dataset labels
  2. lack of generalization in state-of-the-art models
  3. lack of good results in classification as compared to regression. 
   
    To address these problems, we improved upon exisitng models in these ways :  
    1. We first performed the gender classification and used that gender embedding as a prior to the age classification which helped us improve our results.  
    2. In order to generalize our datasets, we performed a model ensemble across multiple datasets and this made our model much more robust.
    3. We have use a new ensembling technique where more weight is given to age clusters grouped together by inversely weighting difference between predicted ages.

    For our age classification tasks, after extensive experimentation we have use the following models
    1. [VisualizingNDF](https://github.com/Nicholasli1995/VisualizingNDF) trained on CACD

    2. [VisualizingNDF](https://github.com/Nicholasli1995/VisualizingNDF) trained on CACD and finetuned on WIKI and UTKface
    3. [VisualizingNDF](https://github.com/Nicholasli1995/VisualizingNDF) trained on CACD and finetuned on WIKI
    4. [VGGFace](https://github.com/rcmalli/keras-vggface) trained on IMDB
### Age
| Model                   | MSE       | RMSE     | R-square  | MAE      |
| ----------------------- | --------- | -------- | --------- | -------- |
| Deep Face (Retina Face) | 332.0787  | 18.223   | \-7.78913 | 14.3249  |
| Deep Face (Opencv)      | 332.8555  | 18.24432 | \-5.4156  | 14.3404  |
| Deep Face (SSD)         | 326.2908  | 18.06352 | \-6.4814  | 14.17737 |
| InsightFace             | 424.6219  | 20.60635 | \-0.46678 | 15.8082  |
| FaceLib                 | 211.54167 | 14.5444  | 0.286898  | 10.09992 |


# File Structure
```bash
├── age_gender_prediction/
├── Deblur/
├── Denoising/
├── ObjDet/ 
├── Super_Resolution/
├── imgs/
├── README.md
├── README.md
```

## Adience Dataset

## Self Trained Models

|                    | epochs | LR            | Val\_acc | test-acc-utk | test-acc-adience | Details                                      | Weight Filename              |
| ------------------ | ------ | ------------- | -------- | ------------ | ---------------- | -------------------------------------------- | ---------------------------- |
| resnet\_gender\_v1 | 10     |               | 93.7     | 93.6         | ~48              |                                              |                              |
|                    |        |               |          |              |                  |                                              |                              |
| VGG-Face-Gender    | 250    | 1.00E-03      | 91.15    | 90.71        | 82.62            | (imagenet weights)                           | gender\_model\_weights.h5    |
|                    | 400    | 1e-4(LRSched) | 92.92    | 91.39        | 74.9             | (imdb-gender-weights)                        | gender\_model\_weights\_1.h5 |
| VGG-Face-Age       |        |               |          |              |                  | (imagenet weights)(10 classes)               |                              |
| IMDB-gender-eval   | 100    | 1.00E-03      | 97.65    |              |                  | reproducing gender model results of vgg face | IMDB-gender-eval.h5          |