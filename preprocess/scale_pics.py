#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import glob

files_paths = glob.glob('emoji_challenge/*/*.jpg')
files_paths = [file for file in files_paths if 'problematic' not in file ]

# This function crops the image to be square with its dimensions min(height, width) x min(height, width)
def crop_img(img): 
    center = (img.shape[0]//2, img.shape[1]//2)
    half_len = min(center[0], center[1])
    img = img[center[0]-half_len:center[0]+half_len, center[1]-half_len:center[1]+half_len]
    return img

# This function zero-pads the image on the top+bottom/left+right so that it's a square
def zero_pad_img(img):
    height, width, channels = img.shape
    if width > height:
        pad_amt = (width - height) // 2
        img = cv2.copyMakeBorder(img, pad_amt, pad_amt, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        pad_amt = (height - width) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad_amt, pad_amt, cv2.BORDER_CONSTANT, value=0)
    return img

if __name__ == '__main__':
    if not os.path.exists('resized_emoji_challenge'):
        os.makedirs('resized_emoji_challenge')
    for file in files_paths:
        relative_path = os.path.join(*(file.split(os.path.sep)[1:]))
        folder, filename = os.path.split(relative_path)
        if not os.path.exists(os.path.join('resized_emoji_challenge', folder)):
            os.makedirs(os.path.join('resized_emoji_challenge', folder))
        image = cv2.imread(file)
        image = zero_pad_img(image)
        image = cv2.resize(image, (256, 256))
        cv2.imwrite(os.path.join('resized_emoji_challenge/', folder, filename), image)

