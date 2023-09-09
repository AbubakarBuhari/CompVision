# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
import cv2
import time
import datetime
#from playsound import playsound

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

#cap = cv2.VideoCapture('rtsp://admin:123456@192.168.1.85/H264?ch=1&subtype=0')
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('rtsp://admin:skybird2022@192.168.1.108/H265?ch=1&subtype=0')
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

img_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
   
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, img_size)
            print("Started Recording!")
            #playsound('sound1.mp3')
            
    elif detection:
        if timer_started:
              if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False 
                timer_started = False
                out.release()
                print('Stop Recording!')
                 #playsound.Exit
                
        else:
            timer_started = True  
            detection_stopped_time = time.time()
            
    if detection:
        out.write(img)
        
            
            

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('NDA PIDS LIVE',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()