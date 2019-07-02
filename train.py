import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils
# import tensorflow.keras.layers


import pandas as pd
import numpy as np

import cv2
from PIL import Image


# tf.data.experimental.TFRecordWriter
def get_data():
    data = pd.read_csv('MNIST_data/train.csv')
    feature = data.iloc[:, 1:].values
    target = data.iloc[:, :1].values.flatten()
    feature=feature.reshape(42000,28,28,1)
    target=utils.to_categorical(target,num_classes=10)
    return feature, target


def build_model(X):
    model = keras.Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]),
        MaxPooling2D(pool_size= (2,2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size= (2,2)),
        Flatten(),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='softmax'),
        Dense(10)
    ])
    
    # optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_absolute_error', 'mean_squared_error','accuracy'])
    return model


def predict(model):
    img=cv2.imread('test.png')
    img_resize=cv2.resize(img,(28,28))
    img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
    img_array=np.array(img_gray)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            img_array[i][j]=255-img_array[i][j]
    result=img_array.reshape(1,28,28,1)
    return model.predict(result)

def show_image():
    feature.reshape(42000,28,28,1)
    a=feature*np.ones((42000,28,28,3))
    a=a.astype(np.uint8 )
    im=Image.fromarray(a[12])
    im.show()
    

def train():
    # tbCallBack = TensorBoard(log_dir='.\logs',  # log 目录
    #     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #     # batch_size=32,     # 用多大量的数据计算直方图
    #     write_graph=True,  # 是否存储网络结构图
    #     write_grads=True, # 是否可视化梯度直方图
    #     write_images=True,# 是否可视化参数
    #     embeddings_freq=0, 
    #     embeddings_layer_names=None, 
    #     embeddings_metadata=None)

    tbCallBack = TensorBoard(log_dir='.\logs')
    feature,target=get_data()
    model=build_model(feature)
    model.fit(feature,target,batch_size = 32, epochs = 3, validation_split = 0.1, callbacks=[tbCallBack])
    model.save('result.h5')

if __name__ == "__main__":
    # model=tf.keras.models.load_model('result.h5')
    # predict(model)
    train()
