import numpy as np
import cv2
import pickle
from PIL import Image


face_detect = cv2.CascadeClassifier('Classifiers/face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

labels = {}
with open("labels.pickle", 'rb') as f:
    org_labels = pickle.load(f)
    labels = {v:k for k,v in org_labels.items()}

cap = cv2.VideoCapture(0)
Alert_text = "Please focus on the road"
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    if faces == ():
    	cv2.putText(img,str(Alert_text), (100,300),font,1,(255,0,0),4, lineType=cv2.LINE_AA)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf>=42 and conf<=85:
            print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,0,0)
            stroke = 2
            cv2.putText(img, name+"--"+str(conf), (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #img_item = "7.jpg"
        #cv2.imwrite(img_item, roi_color)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
        	#print(ex,ey,ew,eh)
        	
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
