import cv2
#import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import random
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import math
from keras.utils.vis_utils import plot_model

def create_model():
    y = Input((128, 128, 3))
    x = Conv2D(32, (5, 5), padding = "same", activation = 'relu', name = "layer1_con1")(y)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = "layer1_pool")(x)
    x = Conv2D(64, (5, 5), padding = "same", activation = 'relu', name = "layer2_con1")(x)
    x = Conv2D(64, (5, 5), padding = "same", activation = 'relu', name = "layer2_con2")(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = "layer2_pool")(x)
    x = Conv2D(128, (5, 5), padding = "same", activation = 'relu', name = "layer3_con1")(x)
    x = Conv2D(128, (5, 5), padding = "same", activation = 'relu', name = "layer3_con2")(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = "layer3_pool")(x)
    x = Conv2D(192, (5, 5), padding = "same", activation = 'relu', name = "layer4_con1")(x)
    x = Conv2D(192, (5, 5), padding = "same", activation = 'relu', name = "layer4_con2")(x)
    x = Conv2D(192, (5, 5), padding = "same", activation = 'relu', name = "layer4_con3")(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = "layer4_pool")(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    out = Dense(10, activation = 'softmax')(x)
    model = Model(inputs = y, outputs = out)
    return model

model = create_model()
plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=True)  