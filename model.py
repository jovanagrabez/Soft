import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
import cv2

def napravi_model():
    
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))


    return model