#!/usr/bin/env python
# coding: utf-8

# # 机器学习工程师纳米学位
# ## capstone
# ## 项目：驾驶员状态检测
#
# 使用深度学习方法检测驾驶员的状态。
#
# 输入：一张彩色图片
# 输出：十种状态的概率
# 状态列表：
#
# - c0: 安全驾驶
# - c1: 右手打字
# - c2: 右手打电话
# - c3: 左手打字
# - c4: 左手打电话
# - c5: 调收音机
# - c6: 喝饮料
# - c7: 拿后面的东西
# - c8: 整理头发和化妆
# - c9: 和其他乘客说话

# In[1]:

# Run some setup code for this notebook.

#import matplotlib.pyplot as plt


# ## 可视化图片

# In[2]:

import cv2
#import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import random

train_data_path = 'datasets/train'
validation_data_path = "datasets/validation"
test_data_path = 'datasets/test'

# for i in range(10):
    # plt.subplot(5, 2, i+1)
    # title = "c{}".format(i)
    # driver_pic = [f for f in listdir(join(train_data_path, title))]
    # img = cv2.imread(join(train_data_path, title, random.choice(driver_pic)))
    # plt.title(title)
    # plt.imshow(img[:,:,::-1]) # 显示图片
    # plt.axis('off') # 不显示坐标轴
# plt.show()


# ## 数据分为训练集和验证集

# In[6]:

from os import rename, mkdir
from os.path import isdir

#print(join("datasets", "validation"))
if not isdir(validation_data_path):
    mkdir(validation_data_path)
for i in range(10):
    if not isdir(join(validation_data_path, "c{}".format(i))):
        mkdir(join(validation_data_path, "c{}".format(i)))
        driver_pic = [f for f in listdir(join(train_data_path, "c{}".format(i)))]
        print(driver_pic)
        val_pic = random.sample(driver_pic, int(len(driver_pic) * 0.2))
        for pic in val_pic:
            rename(join(train_data_path, "c{}".format(i), pic), join(validation_data_path, "c{}".format(i), pic))


# In[21]:

import numpy as np
from tqdm import tqdm

width, height, n_class = 128, 128, 10
train_len = 19150
val_len = 3274

def generate_datasets(X_len, data_path):
    X_gen = np.zeros((X_len, height, width, 3), dtype=np.float32)
    y_gen = np.zeros((X_len, n_class), dtype=np.uint8)
    index = 0
    for i in tqdm(range(10)):
        path = join(data_path, "c{}".format(i))
        driver_pic = [join(path, f) for f in listdir(path)]
        #print(driver_pic)
        for pic in driver_pic:
            print(index)
            X_gen[index] = cv2.resize(cv2.imread(pic), (128, 128))
            y_gen[index][i] = 1
            index += 1
    return X_gen, y_gen

X_train, y_train = generate_datasets(train_len, train_data_path)
X_val, y_val = generate_datasets(val_len, validation_data_path)


# ## 构建trainning data 和 validation data生成器

# In[30]:

from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import math

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
    x = Dense(100,activation='relu')(x)#yzk add
    x = Dropout(0.5)(x)
    out = Dense(10, activation = 'softmax')(x)
    model = Model(inputs = y, outputs = out)
    return model

model = create_model()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

print('traning...')
#board = TensorBoard(log_dir = './logs', histogram_freq=0)
hist = model.fit(X_train, y_train, epochs = 9, validation_data = (X_val, y_val), shuffle = True,
          batch_size = 256, verbose = 1)

with open('fit_log.txt','w') as f:
    f.write(str(hist.history))

print("saving...")
model.save('district_driver.h5')
#print('test...')
#eva = model.evaluate(X_test, [i for i in y_test])
#print(eva)
#print('test end')

