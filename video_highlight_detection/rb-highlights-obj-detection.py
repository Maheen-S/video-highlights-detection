#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
import torch
from pathlib import Path


# In[2]:


get_ipython().run_line_magic('pip', 'install ultralytics')
import ultralytics
ultralytics.checks()


# In[3]:


from ultralytics import YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model


# In[ ]:





# In[4]:


def get_video_files(main_dir, extensions=['.mp4', '.avi', '.mkv']):
    video_files = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    return video_files


# In[5]:


main_dir = '/kaggle/input/sample-cricket-video-clips'

video_files = get_video_files(main_dir)
single_video_path = video_files[0]
single_video_path


# In[6]:


video_frames = {}

def extract_frames(video_path, output_dir):
    video_name = os.path.basename(video_path).split('.')[0]
    video_frames[video_name] = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        frame_path = os.path.join(output_dir, f"frame{count}.jpg")
        cv2.imwrite(frame_path, image)
        video_frames[video_name].append(frame_path)
        success, image = vidcap.read()
        count += 1


# In[8]:


output_dir = os.path.join('/kaggle/working/', os.path.basename(single_video_path).split('.')[0])
extract_frames(single_video_path, output_dir)


# In[9]:


# img_dir = '/kaggle/working/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020'


# In[10]:


img_dir


# In[11]:


output_dir


# In[12]:


output_output_dir = '/kaggle/working/detected_images'
os.makedirs(output_output_dir, exist_ok=True)


# In[ ]:


for img_path in Path(output_dir).glob('*.jpg'):  
    # Perform inference
    results = model(img_path)

    # Save detected image with bounding boxes
    save_path = os.path.join(output_output_dir, f'detected_{img_path.stem}.jpg')
    results.save(save_path)

    print(f"Saved detected image: {save_path}")


# In[ ]:





# In[ ]:


import random
from IPython.display import Image, display

parent_dir = '/kaggle/working/runs/detect'

image_paths = []
for exp_dir in os.listdir(parent_dir):
    exp_path = os.path.join(parent_dir, exp_dir)
    if os.path.isdir(exp_path):
        # List all files in the exp directory
        files = [f for f in os.listdir(exp_path) if f.endswith('.jpg')]
        for file in files:
            image_paths.append(os.path.join(exp_path, file))

num_images_to_display = min(15, len(image_paths))

random_images = random.sample(image_paths, num_images_to_display)

for img_path in random_images:
    display(Image(filename=img_path))


# In[ ]:





# In[ ]:


video_files = get_video_files(main_dir)
print(len(video_files))

if video_files:
    video_path = video_files[5]
    output_video_path = '/kaggle/working/video.mp4'
#     print(len(video_files))
    process_video(video_path, output_video_path, labelsss, model, fps=30)
else:
    print("No video files found"), output_video_path, labels, model, fps=30):


# In[ ]:


def process_video(video_path, output_video_path, labels, model, fps=30):
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    if not success:
        print("Failed to open video")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while success:
#         label = predict_frame_func(frame, model, labels)
#         if label in labels:
#             frame = label_overlay(frame, label)
        out.write(frame)
        success, frame = vidcap.read()
        frame_count += 1

    vidcap.release()
    out.release()
    print(f"Processed {frame_count} frames.")

