# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:31:46 2022

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:25:25 2021

@author: USER
"""

import numpy as np
import cv2
#from scipy.misc import imresize
import time
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

def dsc(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def dice_loss(y_true, y_pred):
        return 1.0 - dsc(y_true, y_pred)


def IOU(y_true, y_pred):

        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
    
        thresh = 0.5

        y_true = K.cast(K.greater_equal(y_true, thresh), 'float32')
        y_pred = K.cast(K.greater_equal(y_pred, thresh), 'float32')

        union = K.sum(K.maximum(y_true, y_pred)) + K.epsilon()
        intersection = K.sum(K.minimum(y_true, y_pred)) + K.epsilon()

        iou = intersection/union

        return iou


model = load_model('LLDNet.h5',custom_objects={'dice_loss':dice_loss,'IOU':IOU,'dsc':dsc,'precision_m':precision_m, 'recall_m':recall_m, 'f1_m':f1_m})





video_capture = cv2.VideoCapture("j.mp4")

video_capture.set(3, 640)
video_capture.set(4, 480)
fps = video_capture.get(cv2.CAP_PROP_FPS)


     
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('j.avi' , fourcc , 20 , (512,512))
count = 0




    
while(video_capture.isOpened()):
  

 

    # Capture the frames
   ret,image = video_capture.read()
   if not ret:
       print("Can't receive frame (stream end?). Exiting ...")
       break
   
   if ret:
           small_img = cv2.resize(image, (512,512))
           small_img = np.array(small_img)
         
           small_img = small_img[None,:,:,:]
          
           prediction = model.predict(small_img)[0] * 255
           crack_image = cv2.resize(prediction, (512,512))
           b, g, r = cv2.split(crack_image)
           z = np.zeros_like(g)
           crack_image = cv2.merge((b, z, z))

           crack_image = crack_image.astype(np.uint8)
           image2=image
           image2=cv2.resize(image2, (512,512))

           result = cv2.addWeighted(image2, 1, crack_image, 1,0)
          
           
           #result=cv2.resize(image, (1080,1920))
           result = result.astype(np.uint8) 
           
           out.write(result)
           cv2.imshow('r',result)
  
  
           if cv2.waitKey(1) & 0xFF == ord('q'):

                 break
       
             
    
       

   
   
    

video_capture.release()
out.release()
cv2.destroyAllWindows()
