#!/usr/bin/env python
# coding: utf-8


import sys
import dlib
import glob
import os


detector = dlib.get_frontal_face_detector()

def print_detections(file): 
    img = dlib.load_rgb_image(file)
    dets, scores, idx = detector.run(img, 1, -1)
    return scores


file_paths = files_paths = glob.glob('resized_emoji_challenge_128_faces/*/*.jpg')
for file in files_paths:
    relative_path = os.path.join(*(file.split(os.path.sep)[1:]))
    scores = print_detections(file)
    if any([score < 0 for score in scores]): 
        print(relative_path)
        os.remove('resized_emoji_challenge_128_faces/' + relative_path)

