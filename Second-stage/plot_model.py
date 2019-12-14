# from google.colab import drive
import cv2
import os
from keras.utils import to_categorical
from keras.utils import plot_model
import numpy as np

from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input
from keras.models import Sequential
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.layers import Input
from keras import backend as K
# drive.mount('/content/drive/')
# !ls'/content/drive/My Drive'

from keras.models import Model

def load_light_data(kind, methor):
    # 人脸转换成100x100的大小
    # 如果是谷歌银盘，把methor 设置成google_drive，同时把r'./'去掉.
    # 给imread前加 '/'
    if methor == 'google_drive':
        Directory_name = 'content/drive/My Drive/myface/' + kind + '/'
    else:
        Directory_name = 'myface/' + kind + '/'
    X_train = []
    y_train = []
    for filename in os.listdir(r'./' + Directory_name):
        X_train.append(cv2.imread(Directory_name + '/' + filename))
        y_train.append(int(filename[0]))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = to_categorical(y_train)  # one-hot

    state = np.random.get_state()  # 打乱数据集
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    # print(y_train[:50])
    return X_train, y_train


def define_model():
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     activation='relu',
                     strides=(2, 2),
                     input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                      activation='relu',
                      padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2056, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1028, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


def get_row_col(num_pic):
    squr = num_pic ** 0.5   # 特征图开方
    row = round(squr)       # 四舍五入
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    print('img_batch',img_batch.shape)  # (1, 10, 10, 384)
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)    # (10, 10, 384)

    feature_map_combination = []
    plt.figure()
    num_pic = feature_map.shape[2]  # 得到特征图的个数
    row, col = get_row_col(num_pic)     # num_pic特征图的个数
    print('num_pic',num_pic)
    print('row,col',row,col)
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]    # 高和宽全部,i是通道
        feature_map_combination.append(feature_map_split)   # 方便叠加
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')
    plt.savefig('feature_map.png')  # 保存图片
    plt.show()
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")  # 保存图片



if __name__ == '__main__':
    X_train, y_train = load_light_data('train', 'normal')
    X_test, y_test = load_light_data('test', 'normal')
    model = define_model()
    model = Model(inputs = model.input,outputs = model.get_layer('conv2d_2').output)
    for i in range(len(model.layers)):
        print(model.get_layer(index=i).output)  # 打印层名
    img_batch = np.expand_dims(X_train[0], axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    visualize_feature_map(conv_img)
