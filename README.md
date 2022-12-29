# Web based object detection with django and pytorch #

Simple and slow object detection app integrating django and pytorch. Detection performed using pytorch pretrained implementation of FASTERRCNN_RESNET50_FPN.

This app is deployed in the local webserver and can be opened from most browsers.

# Steps to use

1.- Download this repository

2.- Create your virtual enviroment and install requirements
```bash
pip install -r requirements
```
3.- Run the App
```bash
python manage.py runserver
```
4.- Open any browser and go to:
```bash
http://localhost:8000/
```
# Expected Results

![Captura de pantalla (1154)](https://user-images.githubusercontent.com/96380180/209953635-54cafe8d-5bb2-4baf-ba5a-22b74b146bf8.png)

# Acknowledgements

Borrowed some ideas from: https://github.com/tranleanh/yolov3-django-streaming and https://github.com/stefanbschneider/pytorch-django

