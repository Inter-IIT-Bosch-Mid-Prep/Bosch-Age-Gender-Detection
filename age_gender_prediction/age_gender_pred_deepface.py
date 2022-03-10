from deepface import DeepFace
from matplotlib import pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def find_agebucket(age):
    if int(age)>=13 and int(age)<=17:
        return '13-17years'
    elif int(age)>17 and int(age)<=24:
        return '18-24years'
    elif int(age)>24 and int(age)<=34:
        return '25-34years'
    elif int(age)>34 and int(age)<=44:
        return '35-44years'
    elif int(age)>44 and int(age)<=54:
        return '45-54years'
    elif int(age)>54 and int(age)<=64:
        return '55-64years'
    elif int(age)>64:
        return 'above 65years'
    else:
        return 'NA'

def calculate_gender(image):
    person={}
    path = image_path
    try:
        #img_arr=cv2.imread(path)
        img_arr = img_arr[:,:,[2,1,0]]
        ## get gender
        response=DeepFace.analyze(img_arr,actions=["gender","age"],enforce_detection=False)
        gender = response['gender']
        age = response['age']
        ## Bucket the age
        agebucket = find_agebucket(age)
        ## store in dictionary
        person[image_path] = {'gender':gender, 'age': age, 'agebucket': agebucket}
    except:
        person[image_path]='NA' ## If the image is not a front facing image
    return gender , age , agebucket