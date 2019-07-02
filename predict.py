import tensorflow as tf
import cv2
import numpy as np


def predict(model,filename):
    img=cv2.imread(filename)
    img_resize=cv2.resize(img,(28,28))
    img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
    img_array=np.array(img_gray)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            img_array[i][j]=255-img_array[i][j]
    result=img_array.reshape(1,28,28,1)
    return model.predict(result)

if __name__ == "__main__":
    model=tf.keras.models.load_model('result.h5')
    print('This number is {}'.format(predict(model,'test.png').argmax()))