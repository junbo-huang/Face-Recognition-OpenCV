import cv2
import os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'dataSet'

def get_images_and_labels(path):
     file_paths = [os.path.join(path, f) for f in os.listdir(path)]
     #print(file_paths)
     # images will contains face images
     images = []
     # labels will contains the label that is assigned to the image
     labels = []
     for file_path in file_paths:
         # Read the image and convert to grayscale
         image_pil = Image.open(file_path).convert('L')
         # Convert the image format into numpy array
         # Get the label of the image
         ID = int(os.path.split(file_path)[-1].split(".")[1])
         print(ID)
         size = (400, 400)
         final_image = image_pil.resize(size, Image.ANTIALIAS)

         image = np.array(final_image, 'uint8')
         faces = faceCascade.detectMultiScale(image)
         # If face is detected, append the face to images and the label to labels
         for (x, y, w, h) in faces:
             images.append(image[y:y +h, x:x+w])
             labels.append(ID)
             cv2.imshow("Adding faces to traning set...", image[y:y+h, x:x+w])
             cv2.waitKey(10)
     # return the images list and labels list
     return images, labels

 
images, labels = get_images_and_labels(path)
print(images, labels)

recognizer.train(images, np.array(labels))
recognizer.save('trainer_2.yml')
cv2.destroyAllWindows()
