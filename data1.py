import numpy as np
import cv2
from PIL import Image

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


name = str(input("please enter the name of the user: "))
#user_id=input('please enter user id: ')
sample = 0
while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detect.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		sample += 1
		cv2.imwrite("images/"+name+str(sample)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		cv2.waitKey(100)
		cv2.imshow('image', img)
	cv2.waitKey(1)
	if (sample>15):
		break

cap.release()
cv2.destroyAllWindows()
