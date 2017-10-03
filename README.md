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

We will step through a brief overview of how to find faces and eyes in images using **Python 3.5**
###
![][image10]
##### *(A tribute to the father of Python)*

###

###

We will leverage the following Python Modules:
- OpenCV (cv2)
- PIL
- pylab
- matplotlib.pyplot
- os

OpenCV will do the majority of the heavy lifting. I've adopted an example from the documentation, [here.](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection)

I would suggest you check out the [documentation](http://docs.opencv.org/master/d6/d00/tutorial_py_root.html) for implementation: 
###

We first need to set the OpenCV classifiers to `haarcascade`. To learn more about Cascade Classifier Training, [check this out](http://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html). For a step deeper, here is some [background on the Haar Wavelet.](https://en.wikipedia.org/wiki/Haar_wavelet)

These classifiers comes pre-trained. One for the face:
```
face_cascade = cv2.CascadeClassifier('/Users/UserName/anaconda/envs/py35/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
```
One for the eyes:
```
eye_cascade = cv2.CascadeClassifier('/Users/UserName/anaconda/envs/py35/share/OpenCV/haarcascades/haarcascade_eye.xml')
```        
Using Anaconda, my XML files were located at the above path. If you're having trouble finding the files, search your computer for `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` and alter the above path to point to them.

We're going to use `matplotlib.pyplot` to open a figure and title it:

```
plt.figure('Facial Recognition')
plt.suptitle('Facial Recognition', size=20)
```

Select the image we want to detect the faces and eyes in. I'm going to use an old photo of Jobs and Wozniak to demonstrate:

```
imageName = 'apple.jpg'
```     
![][image1]

###
We're now going to plot the original image:
```
img = array(Image.open(imageName))
upperLeft = plt.subplot(221)
```
###
Side note: I always wondered what the numbers (221,222,223,224) in `pyplot` dictated -- it stands for a `2x2` table of photos, and the third digit dictates the position of the image. Seems obvious now...For example, you could plot 16 photos in (441,442,443,...,).

Anyway, back to it. 

We're going to now harness `PIL's` `Image` to grayscale the image:

```
gray = array(Image.open(imageName).convert('L'))
```
See what it looks like with:
```
plt.imshow(gray, cmap='gray')
```
We'll now plot those and begin with the face detection, and draw a red rectangle around the face:
```
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
for (x,y,w,h) in faces:
    face = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
```
###
This yields the original image, the grayscale image for analysis, and the detected faces plotted in our figure:

###
###

![][image2]
###
Next is to find the eyes, and draw green rectangles around them. This can get a bit tricky, and is often where most inaccuracies occur. We find the eyes with:
```
eyes = eye_cascade.detectMultiScale(roi_gray)

for (ex,ey,ew,eh) in eyes:
    eye = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
```
These pre-trained classifiers make things very easy on us, but have their issues (especially for the eyes). I suggest digging in deeper to these functions to see what is going on behind the scenes.

Of the images I tested, the classifier did best on Guido's picture to categorize his face and eyes accurately:
###
![][image10]
###
The classifier will often incorrectly classify the mouth or the nose as an eye (assumingly due to the pixel intensity differential that is similar to the eyes). For example, this image of Sheryl Sandberg:
###
![][image12]
###
Or this image of Edison:
###
![][image6]
###

As mentioned before, if an image does not have visible faces in it, the Haar features will not be detected. The `try except` statement helps with this error handling.  Naturally, our classifier can't find faces in this image of the Chrysler building, so it just outputs the first two images of the analysis:
###
![][image4]
###
Also, if there are too many faces in the image, this can also be an issue. As seen here in the team from Willow Garage:
###
![][image14]
###

###

As you can tell, this classifer could be improved when it comes to eye classification. However, for a pre-built library, I'm happy with what it can do.



### Next steps: Implementing this openCV library on the Raspberry Pi Zero W... then perhaps a *smile detector!*

