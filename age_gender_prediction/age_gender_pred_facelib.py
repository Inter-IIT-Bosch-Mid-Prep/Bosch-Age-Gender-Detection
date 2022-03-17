from facelib import FaceDetector, AgeGenderEstimator
import matplotlib.pyplot as plt
import torch
import numpy as np

# age_gender_detector = AgeGenderEstimator()



def find_agebucket(age):

    if int(age)>=0 and int(age)<13:
        return '0-12years', 0, 12 
    elif int(age)>=13 and int(age)<=17:
        return '13-17years', 13, 17
    elif int(age)>17 and int(age)<=24:
        return '18-24years', 18, 24
    elif int(age)>24 and int(age)<=34:
        return '25-34years', 25, 34
    elif int(age)>34 and int(age)<=44:
        return '35-44years', 35, 44
    elif int(age)>44 and int(age)<=54:
        return '45-54years', 45, 54
    elif int(age)>54 and int(age)<=64:
        return '55-64years', 55, 64
    elif int(age)>64 and int(age)<=74:
        return '65-74years', 65, 74
    elif int(age)>74 and int(age)<=84:
        return '75-84years', 75, 84
    elif int(age)>84 and int(age)<=94:
        return '85-94years', 85, 94
    elif int(age)>94 and int(age)<=104:
        return '95-104years', 95, 104
    elif int(age)>104 and int(age)<=114:
        return '105-114years', 105, 114
    else:
        return "NA",0,0
    
def calculate_gender(image_path, age_gender_detector, device):
    person={}
    image = image_path
#     try:
#         img_arr=cv2.imread(path)
#         img_arr = img_arr[:,:,[2,1,0]]
        ## get gender
    #image = plt.imread(path)
    image = [image]
    image = torch.tensor(np.array(image), device=device)
    genders, ages = age_gender_detector.detect(image)
    gender = genders[0]
    age = ages[0]
    ## Bucket the age
    agebucket = find_agebucket(age)
    ## store in dictionary
#    person[image_path] = {'gender':gender, 'age': age, 'agebucket': agebucket}
#     except:
#         person[image_path]='NA' ## If the image is not a front facing image
    return gender , age , agebucket


