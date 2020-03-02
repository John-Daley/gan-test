from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np 
from PIL import Image
import os

# preview image frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4 
SAVE_FREQ = 100

# SIZE VECTOR GENERATE IMAGES FROM 
NOISE_SIZE = 100

#CONFIG
EPOCHS = 10000 
BATCH_SIZE = 32

#IMAGES SHOULD ALWAYS BE SQUARE SIZED FOR THIS
GENERATE_RES = 3
IMAGE_SIZE = 128 # ROWS/COLS

IMAGE_CHANNELS = 3

training_data = np.load('cubism_data.npy')

def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3, strides=2, input_shape = image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)

    