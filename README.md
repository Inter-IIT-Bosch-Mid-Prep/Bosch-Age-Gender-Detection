# Bosch-Age-Gender-Detection-IITKGP

Run the following commands to detect gender and age.

```
git clone -b image_gender_age_detection https://github.com/Inter-IIT-Bosch-Mid-Prep/Bosch-Age-Gender-Detection-IITKGP.git
```

```
cd {Bosch-Age-Gender-Detection-IITKGP}
python detect.py --weights "weights path of pretrained yolov5"  --weights_gan "weights path of pretrained edsr" --image "image path" --output_folder "folder were output image is to be stored" 
```
