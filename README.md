# Real_Time_Mask_Detector
Real Time Medical Mask Detector

It's simple project that detect the medical mask in realTime camera input using Keras - OpenCv - Facerecognition Library.

![alt img](./demo.gif)<br>

# Summary

There are 2 different CNN models in this project. One of them comes from FaceRecognition library which detects the face and returns it's position. Other one is special trained model to classify image as with_mask or without_mask. An Image which we get from camera by using openCv, firstly enters the face detection model to get face position. Then, crops the area from borders and enters to the second model to get mask existence.

# Dataset and Pretrained Model

The dataset we used consists of 1,376 images with 2 different classes. with_mask and without_mask. There are 690 images for each of them. Dataset is artificially made by Prajna Bhandary **[Linkedin](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/)**. Biggest Thanks. You can Download the entire dataset from **[HERE](https://drive.google.com/drive/folders/1cHLb3oX7gLRkrrnIMGbduVvFF2msok6z?usp=sharing/)**

You can get pretrained model **[HERE](https://drive.google.com/file/d/1ZnNhitQjHcs0c-Ir_4KhJ3h96u6fC-7W/view?usp=sharing/)**


# Dependencies

1-Face recognition library - `pip3 install face_recognition`

2-Keras, OpenCv, Matplotlib (There are tons of "how to install" tutorials out there.)

# How to Run

If you want to train your own model, you have to download dataset and locate into directory of train.py. If you want to use pretrained model, download model and locate it into same folder.

1- `python3 train.py`

2- `python3 run.py`

# Related Works -Source

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
https://towardsdatascience.com/real-time-face-mask-detector-with-tensorflow-keras-and-opencv-38b552660b64
