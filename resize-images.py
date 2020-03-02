#this code is being adapted from the following tutorial, at this link https://towardsdatascience.com/generating-modern-arts-using-generative-adversarial-network-gan-on-spell-39f67f83c7b4


import os
import numpy as np 
from PIL import Image


IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = '/dataset'

images_path = IMAGE_DIR

training_data = []

print('resizing images to 128x128')

for file_name in os.listdir(images_path):
    path = os.path.join(images_path, file_name)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
    training_data.append(np.asarray(image))
training_data = np.reshape(training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data /127.5 -1

print(' saving resized files to cubism_data.npy...')
np.save('cubism_data.npy, training_data')
