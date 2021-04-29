# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:41:56 2021

@author: Administrator
"""
from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib

root = tkinter.Tk()
root.withdraw()

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

model = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vid_source = cv2.VideoCapture(0)

text_dict = {0:'Mask ON',1:'No Mask'}
rect_color_dict = {0:(0,255,0), 1:(0,0,255)}

SUBJECT = "Subject"
TEXT = "One Visitor violated the Face Mask Policy. See in the camera to recognise the user.";

while(True):
    ret,img = vid_source.read()
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR)
    faces = faceNet.detectMutltiScale(grayscale_img,1.3,5)
    
    for(x,y,w,h) in faces:
        face_img = grayscale_img[y:u+w,x:x+w]
        resized_img = cv2.resize(face_img,(224,224))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,224,224,1))
        result=model.predict(reshaped_img)
        
        label = np.argmax(result,axis = 1)[0]
        
        cv2.rectangle(img, (x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img, (x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img, text_dict[label], (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        
        if(label == 1):
            messagebox.showwarning("Warning", "Access Denied. Please wear a Face Mask")
            
            message = 'Subject : {}\n\n{}'.format(SUBJECT,TEXT)
            mail = smtplib.SMTP('smtp.gmail.com',587)
            mail.ehlo()
            mail.starttls()
            mail.login('vsoftnotification@gmail.com', 'mahalakshmi123')
            mail.sendmail('vsoftnotification@gmail.com', 'mahalakshmithirumurthy@gmail.com', message);
            mail.close()
        else:
            pass
            break
    cv2.imshow('LIVE Video Feed',img)
    key = cv2.waitKey(1)
    
    if(key == 27):
        break
cv2.destroyAllWindows()
source.release()