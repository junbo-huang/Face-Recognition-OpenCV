import numpy as np
import cv2
import winsound


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = cv2.VideoCapture(0)
Alert_text = "please focus on the road ahead"
Alert_text2 = "Stop driving and take a break"
font = cv2.FONT_HERSHEY_SIMPLEX
timer = 0

while True:
    ret, detector = camera.read()
    gray = cv2.cvtColor(detector, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    if str(faces) == "()":
        timer += 1
        if timer > 25:
            #cv2.putText(detector,str(Alert_text), (70,300),font,1,(0,0,255),4, lineType=cv2.LINE_AA)
            winsound.PlaySound('SystemAsterisk', winsound.SND_ALIAS)
        if timer > 50:
            timer = 0
    else:
        timer = 0

    for (x,y,w,h) in faces:
        #cv2.rectangle(detector,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = detector[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)

        #if str(eyes) == "()":
            #timer += 1
            #print(timer)
            #cv2.putText(detector,str(Alert_text2), (80,120),font,1,(255,100,255),4, lineType=cv2.LINE_AA)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

    #cv2.imshow('image',detector)
    k = cv2.waitKey(30) & 0xff
    if k == ord('x'):
        break


camera.release()
cv2.destroyAllWindows()
