#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:15:58 2020

@author: berk
"""
import cv2
from tensorflow.keras.models import load_model
import face_recognition 
import numpy as np


#------------------------------------------------------
#This section for reduce some errors related to GPU allocation on my system.
#it may not neccesary for yours. If it is not, removing this part may increase the performance.
from tensorflow import Session,ConfigProto
from keras.backend.tensorflow_backend import set_session
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(Session(config=config))
#--------------------------------------------------------

width=150
heigh=150


#load trained model from current directory
model = load_model('model.h5') 

cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    
    result = face_recognition.face_locations(frame)

    if result != []:
        for person in result:
            #MTCNN returns coordinates like (y0,x0,y1,x1) 
            #Crop the face and send to my model
            #If face located in edge of the frame exception will occur. 
            try:
                #it`s cropped 80 pixel wider than original face detection coordinates.
                croppedImg = frame[person[0]-40:person[2]+40,person[3]-40:person[1]+40]
                #resize image 150x150 px
                croppedImg=cv2.resize(croppedImg,(width,heigh),interpolation = cv2.INTER_AREA)
                result = model.predict(croppedImg[None])
            except:
                #if face is to close to border of frame, it cannot perform the crop oparation. Therefore, we throw an exception.  
                print("You are too far from center")
            cv2.rectangle(frame,(person[3],person[0]),(person[1],person[2]),
                          (0,155,255),
                          2)
            print(result)
            #look the index of maximum prediction, put text into frame
            if np.argmax(result)== 0:
                cv2.putText(frame,"Mask Off",(person[3],person[0]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            else:
                cv2.putText(frame,"Mask On",(person[3],person[0]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()



