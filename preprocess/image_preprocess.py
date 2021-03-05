#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import glob
import matplotlib.pyplot as plt
import os
import face_recognition

# In[16]:


# DEFINE CONSTANTS
SIZE = (128, 128)
SAVE_DIRECTORY = 'resized_emoji_challenge_128_faces'


# In[17]:


files_paths = glob.glob('emoji_challenge/*/*.jpg')
print(f'Total: {len(files_paths)}')
files_paths = [file for file in files_paths if 'problematic' not in file]
print(f'Excluding problematic: {len(files_paths)}')

EXCLUDE_DIRS = ['eye_roll', 'here_we_go', 'surprise', 'angry']
files_paths = [file for file in files_paths if not any(x in file for x in EXCLUDE_DIRS)]


for folder in os.listdir('emoji_challenge'):
    if os.path.isdir(os.path.join('emoji_challenge', folder)):
        print(f'{folder}: {len(os.listdir(os.path.join("emoji_challenge", folder)))}')


# This function crops the image to be square with its dimensions min(height, width) x min(height, width)
def crop_img(img): 
    center = (img.shape[0]//2, img.shape[1]//2)
    half_len = min(center[0], center[1])
    img = img[center[0]-half_len:center[0]+half_len, center[1]-half_len:center[1]+half_len]
    return img


# In[21]:


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

# ## Remove sketched emojies
# Many of the images are in the shape of a square and have a picture of a person on the left and a picture of a hand-drawn emoji on the right. We don't care about the emoji on the right, so we want to remove it. 



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


# Returns tuple of location of the largest face in the image
def find_face_loc(locations):
    max_area = 0
    max_loc = (0, 0, 0, 0)
    for tup in locations: 
        top, right, bottom, left = tup
        area = (bottom - top) * (right - left)
        max_area = area if area > max_area else max_area
        max_loc = tup if area == max_area else max_loc
    return max_loc


# In[31]:


# Returns cropped image of largest face in the photo (or uncropped image if no face found)
def crop_face(img): 
    locations = face_recognition.face_locations(img)
    face_loc = find_face_loc(locations)
    if face_loc == (0, 0, 0, 0): 
        return None
    top, right, bottom, left = face_loc
    return img[top:bottom, left:right, :]


if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
for file in files_paths:
    relative_path = os.path.join(*(file.split(os.path.sep)[1:]))
    print(relative_path)
    folder, filename = os.path.split(relative_path)
    if not os.path.exists(os.path.join(SAVE_DIRECTORY, folder)):
        os.makedirs(os.path.join(SAVE_DIRECTORY, folder))
    image = cv2.imread(file)
    image = tall_image(image)
    image = crop_face(image)
    if image is None:
        continue
    image = zero_pad_img(image)
    image = cv2.resize(image, SIZE)
    cv2.imwrite(os.path.join(f'{SAVE_DIRECTORY}/', folder, filename), image)
