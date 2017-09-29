import cv2
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import os


def main():

    try:
        face_cascade = cv2.CascadeClassifier('/Users/ChrisErnst/anaconda/envs/py35/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('/Users/ChrisErnst/anaconda/envs/py35/share/OpenCV/haarcascades/haarcascade_eye.xml')
        # If you have trouble finding the xml files, search your computer for 'haarcascade_frontalface_default.xml' and 'haarcascade_eye.xml' and alter the above path
        
        imageDirectory = '/Users/ChrisErnst/Development/Python/computerVision/facialRecognition/images/'
        os.chdir(imageDirectory)
        # Set the directory
        
        plt.figure('Facial Recognition')
        plt.suptitle('Facial Recognition', size=20)
        # Generate a figure named 'Face Recognition' and add a title
        
        imageName = 'apple.jpg'
        # Set the image we want to analyze
        
        img = array(Image.open(imageName))
        upperLeft = plt.subplot(221)
        # The convention of 221 is the 1st image of a 2x2 figure
        plt.axis('off')
        plt.imshow(img)
        upperLeft.set_title('Original Image')
        # Show the original image using PIL and plot using pyplot in the upper left
        
        gray = array(Image.open(imageName).convert('L'))
        upperRight = plt.subplot(222)
        plt.axis('off')
        plt.imshow(gray, cmap='gray')
        upperRight.set_title('Grayscale to Analyze')
        # Convert to gray with PIL and plot using pyplot in the upper right
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            face = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
        lowerLeft = plt.subplot(223)
        plt.axis('off')
        plt.imshow(face)
        lowerLeft.set_title('Face Detection')
        # Find the face in the photo, draw a red rectangle around it, and plot in the lower left
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            eye = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        lowerRight = plt.subplot(224)
        plt.axis('off')
        plt.imshow(eye)
        lowerRight.set_title('Eye Detection')    
        # Find the eyes in the photo, draw green rectangles around them, and plot in the lower right
        
        plt.savefig(imageDirectory + imageName[0:-4] +'faceDetect.jpg')
        # Save the file    
        
    except UnboundLocalError:
        plt.savefig(imageDirectory + imageName[0:-4] +'faceDetect.jpg')
        # Save the file
        print("Make sure there are faces and eyes in your image!")
        # Exception for an image without faces and/or eyes
        
    except FileNotFoundError:
        print("Image not found. Verify the file name and path!")
        # Exception for bad image path
            
main()
    