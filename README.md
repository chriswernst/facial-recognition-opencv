# Facial Recognition

#### Overview

[//]: # (Image References)

[image1]: ./images/apple.jpg
[image2]: ./images/applefaceDetect.jpg
[image3]: ./images/chrysler.jpg
[image4]: ./images/chryslerfaceDetect.jpg
[image5]: ./images/edison.jpg
[image6]: ./images/edisonfaceDetect.jpg
[image7]: ./images/elon.jpg
[image8]: ./images/elonfaceDetect.jpg
[image9]: ./images/guido.jpg
[image10]: ./images/guidofaceDetect.jpg
[image11]: ./images/sandberg.jpg
[image12]: ./images/sandbergfaceDetect.jpg
[image13]: ./images/willow.jpg
[image14]: ./images/willowfaceDetect.jpg

We will step through a brief overview of how to find faces in images using **Python 3.5**
###
![][image10]

*(A tribute to the father of Python)*

###
We will leverage the following Python Modules:
- OpenCV (cv2)
- PIL
- pylab
- matplotlib
- os

OpenCV will do the majority of the heavy lifting. I've adopted [an example from the documentation, here.](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection)

I would suggest you check out the [documentation](http://docs.opencv.org) for implementation: 
###

We first need to set the OpenCV classifiers to `haarcascade`. These are pre-trained classifiers for faces and eyes. To learn more about Cascade Classifier Training, [check this out](http://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html).

One for the face:
```
face_cascade = cv2.CascadeClassifier('/Users/ChrisErnst/anaconda/envs/py35/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
```
One for the eyes:
```
eye_cascade = cv2.CascadeClassifier('/Users/ChrisErnst/anaconda/envs/py35/share/OpenCV/haarcascades/haarcascade_eye.xml')
```        
With Anaconda, my XML files were located at the above path. If you're having trouble finding the files, search your computer for `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` and alter the above path to point to them.

We're going to use `matplotlib.pyplot` to open a figure and title it:

```
plt.figure('Facial Recognition')
plt.suptitle('Facial Recognition', size=20)
```

Select the image we want to detect the faces and eyes in. I'm going to use an old photo of Jobs and Wozniak to demonstrate:

```
imageName = 'apple.jpg'
```     
![][image10]

###
We're now going to plot the original image:
```
img = array(Image.open(imageName))
upperLeft = plt.subplot(221)
```
###
Side note: I always wondered what the numbers (221,222,223,224) in `pyplot` dictated, it finally clicked that it stands for a `2x2` table of photos, and the third digit dictates the position of the image. For example, you could plot 16 photos in (441,442,443,...,).

Anyway, back to it. 

We're going to now harness `PIL's` `Image` to grayscale the image:

```
gray = array(Image.open(imageName).convert('L'))
```
See what it looks like with:
```
plt.imshow(gray, cmap='gray')
```

### *(README In Progress)*



        
