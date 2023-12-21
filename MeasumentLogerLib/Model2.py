# https://www.youtube.com/watch?v=bte8Er0QhDg
# https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers
# https://www.kaggle.com/datasets/hugovallejo/street-view-house-numbers-svhn-dataset-numpy

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    #model.save('testModel ') not working dont know why

    loss, accuracy = model.evaluate(x_test, y_test)

    img = cv2.imread(r"../frames/0.jpg")
    prediction = model.predict(img)
    print(f"most likely {np.argmax(prediction)}")
