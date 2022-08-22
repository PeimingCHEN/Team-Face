# Team Face
* [Introduction](#introduction)
* [Usage](#usage)
* [Functions](#functions)
* [Tech Stack](#tech-stack)
* [Demo](#demo)
* [Reference](#reference)

## Introduction
TeamFace, an android app, applied the face recognition deep learning algorithm to recognize the entered face and authenticate the user’s identity of the organization. Our app can be widely used in various personnel management application scenarios.

## Usage
* Administrator system & Android APK installation package: click [here](http://39.103.167.15:6789).
* Use the organization code **000** to register as a user of the **Test Group**.
* Click ‘设置’ and follow the instructions to set up the user images.
* Click ‘人脸识别’ to recognize your face and authenticate your identity.

## Functions
* The users can only successfully register and bind to the corresponding organization by the organization code.
* Successfully registered users can log in and out of the app.
* Users can collect and set their faces in the app to authenticate their identities.
* The bound organization will be displayed if the face recognition is successful.
* The administrator can manage user data in the administrator system, such as modifying user information, adding organizations, setting registration codes, etc.

## Tech Stack
* Django
* Python
* Flutter
* Rest_framework
* OpenCV
* Tensorflow
* Nginx
* uWSGI

## Demo
<p align="center">
<img src="demo/login.jpg" alt="login" width="250"/> <img src="demo/signup.jpg" alt="signup" width="250"/> <img src="demo/home.jpg" alt="home" width="250"/><br>
<img src="demo/setting_success.jpg" alt="setting_success" width="250"/> <img src="demo/rec_success.jpg" alt="rec_success" width="250"/> <img src="demo/rec_failed.jpg" alt="rec_failed" width="250"/>
</p>

## Reference
Face-Net: https://github.com/bubbliiiing/facenet-tf2 <br>
Siamese-Net: https://github.com/nicknochnack/FaceRecognition <br>
Deepface: https://github.com/serengil/deepface 
