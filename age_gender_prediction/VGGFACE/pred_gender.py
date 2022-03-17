import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cv2

import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split

from keras import metrics

from keras.models import model_from_json
import matplotlib.pyplot as plt


def gender_prediction(image, model):

  image = np.expand_dims(cv2.resize(image, (224, 224)), axis=0)
  return np.argmax(model(image)[0])






