import cv2
import os
import numpy as np
from PIL import Image 
import pickle
#import winsound

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer_2.yml')
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
path = 'dataSet'

Alert_text = "please focus on the road ahead"
Alert_text2 = "Stop driving and take a break"
font = cv2.FONT_HERSHEY_SIMPLEX
timer = 0

cap = cv2.VideoCapture(0)
while True:
    ret, img =cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    if str(faces) == "()":
        timer += 1
        if timer > 25:
            cv2.putText(img,str(Alert_text), (70,300),font,1,(0,0,255),4, lineType=cv2.LINE_AA)
            #winsound.PlaySound('SystemAsterisk', winsound.SND_ALIAS)
        if timer > 60:
            timer = 0
    else:
        timer = 0

    for (x,y,w,h) in faces:
        ID_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        if (conf>=40 and conf <= 75) and (ID_predicted==1):
            ID_predicted = 'Handsome'
            cv2.putText(img,str(ID_predicted)+"--"+str(conf), (x,y+h),font, 1, (255,0,0),2, cv2.LINE_AA)
        elif (conf>=48 and conf <= 85) and (ID_predicted==2):
            ID_predicted = 'Ugly'
            cv2.putText(img,str(ID_predicted)+"--"+str(conf), (x,y+h),font, 1, (255,255,0),2, cv2.LINE_AA) #Draw the text
        elif (conf>=45 and conf <= 80) and (ID_predicted==3):
            ID_predicted = 'Jason'
            cv2.putText(img,str(ID_predicted)+"--"+str(conf), (x,y+h),font, 1, (0,255,255),2, cv2.LINE_AA)

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if str(eyes) == "()":
            #timer += 1
            #print(timer)
            cv2.putText(img,str(Alert_text2), (80,120),font,1,(255,100,255),4, lineType=cv2.LINE_AA)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)


    
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()









