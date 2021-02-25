#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import glob
import matplotlib.pyplot as plt
import os
import face_recognition

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


files_paths = glob.glob('emoji_challenge/*/*.jpg')
print(f'Total: {len(files_paths)}')
files_paths = [file for file in files_paths if 'problematic' not in file]
print(f'Excluding problematic: {len(files_paths)}')


# In[23]:


for folder in os.listdir('emoji_challenge'):
    if os.path.isdir(os.path.join('emoji_challenge', folder)):
        print(f'{folder}: {len(os.listdir(os.path.join("emoji_challenge", folder)))}')


# ## Cropping and padding
# In[9]:


# This function crops the image to be square with its dimensions min(height, width) x min(height, width)
def crop_img(img): 
    center = (img.shape[0]//2, img.shape[1]//2)
    half_len = min(center[0], center[1])
    img = img[center[0]-half_len:center[0]+half_len, center[1]-half_len:center[1]+half_len]
    return img


# In[10]:


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


# Returns the proportion of an image's height to its width
def dimension_prop(img): 
    height, width, n_channels = img.shape
    return height / width


# In[26]:


# Returns original image if original image is tall, otherwise cuts it in half widthwise
def tall_image(img): 
    img_prop = dimension_prop(img)
    if img_prop < 1.25: # arbitrary boundary, we can try other stuff too
        width = img.shape[1] // 2
        return img[:, :width, :]
    else: 
        return img

if not os.path.exists('resized_emoji_challenge_128'):
    os.makedirs('resized_emoji_challenge_128')
for file in files_paths:
    relative_path = os.path.join(*(file.split(os.path.sep)[1:]))
    folder, filename = os.path.split(relative_path)
    if not os.path.exists(os.path.join('resized_emoji_challenge_128', folder)):
        os.makedirs(os.path.join('resized_emoji_challenge_128', folder))
    image = cv2.imread(file)
    image = tall_image(image)
    if not len(face_recognition.face_locations(image)):
        continue
    image = zero_pad_img(image)
    image = cv2.resize(image, (128, 128))
    cv2.imwrite(os.path.join('resized_emoji_challenge_128/', folder, filename), image)