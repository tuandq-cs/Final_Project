# Face Detection & Tracking & Extract

![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)

This project can **detect** , **track** and **extract** the **optimal** face in multi-target faces (exclude side face and select the optimal face).

## Introduction

* **Dependencies:**
  * Python 3.5+
  * Tensorflow 1.15
  * [**MTCNN**](https://github.com/davidsandberg/facenet/tree/master/src/align)
  * Scikit-learn 0.21
  * Numpy
  * Numba
  * Opencv-python
  * Filterpy

## Run

* Create conda environment with Python 3.6:
  ```
  conda create -n myenv python=3.6 pip
  conda activate myenv
  ```
* Install packages:
  ```
  python3 -m pip install -r requirements.txt
  ```
* To run the python version of the code :

```sh
python3 start.py [-h] [--videos_dir VIDEOS_DIR]
                [--output_images_dir OUTPUT_IMAGES_DIR]
                [--output_videos_dir OUTPUT_VIDEOS_DIR] [--save_videos]
                [--detect_interval DETECT_INTERVAL] [--margin MARGIN]
                [--scale_rate SCALE_RATE] [--show_rate SHOW_RATE]
                [--face_score_threshold FACE_SCORE_THRESHOLD]
                [--face_landmarks] [--no_display]
```

```
optional arguments:
  -h, --help            show this help message and exit
  --videos_dir VIDEOS_DIR
                        Path to the data directory containing aligned your
                        face patches.
  --output_images_dir OUTPUT_IMAGES_DIR
                        Path to save face images
  --output_videos_dir OUTPUT_VIDEOS_DIR
                        Path to save inference video
  --save_videos         Save inference video or not
  --detect_interval DETECT_INTERVAL
                        how many frames to make a detection
  --margin MARGIN       add margin for face
  --scale_rate SCALE_RATE
                        Scale down or enlarge the original video img
  --show_rate SHOW_RATE
                        Scale down or enlarge the imgs drawn by opencv
  --face_score_threshold FACE_SCORE_THRESHOLD
                        The threshold of the extracted faces,range 0<x<=1
  --face_landmarks      Draw five face landmarks on extracted face or not
  --no_display          Display or not
```

* Then you can find  faces extracted stored in the floder **./facepics** .
* If you want to draw 5 face landmarks on the face extracted,you just add the argument **face_landmarks**

```sh
python3 start.py --face_landmarks
```

* If you want to render video, you just add the argument **save_videos**
  ```
  python3 start.py --save_videos
  ```

🚀️ **IMPORTANT:**🚀️  If your are gonna run this source in server. Please make sure that add the argument **--no_display**

```
python3 start.py --no_display
```

## What can this project do?

* You can run it to extract the optimal face for everyone from a lot of videos and use it as a training set for **CNN Training**.
* You can also send the extracted face to the backend for **Face Recognition**.

## Results

![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic4.gif "scene 1")
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic5.jpg "faces extracted")

## Special Thanks to:

* [**experimenting-with-sort**](https://github.com/ZidanMusk/experimenting-with-sort)

## License

MIT LICENSE
