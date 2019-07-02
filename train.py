import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils


import pandas as pd
import numpy as np


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

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_absolute_error', 'mean_squared_error','accuracy'])
    return model


def train():
    tbCallBack = TensorBoard(log_dir='.\logs')
    feature,target=get_data()
    model=build_model(feature)
    model.fit(feature,target,batch_size = 32, epochs = 3, validation_split = 0.1, callbacks=[tbCallBack])
    model.save('result.h5')

if __name__ == "__main__":

    train()
