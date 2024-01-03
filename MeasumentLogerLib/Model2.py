# https://www.youtube.com/watch?v=bte8Er0QhDg
# https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers
# https://www.kaggle.com/datasets/hugovallejo/street-view-house-numbers-svhn-dataset-numpy

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


matplotlib.use('TkAgg')


if __name__ == '__main__':
    #model_name = r"model2.h5"
    model_name = r"model2"
    if not os.path.exists(model_name):
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

        loss, accuracy = model.evaluate(x_test, y_test)

        model.save(model_name) #o formato default de salvar modelos do tensorflow consiste em uma estrutura de arquivos, pesquisar como é a mesma depois
    else:
        print(f'Opening saved model: \'{model_name}\'')

    loaded_model = tf.keras.models.load_model(model_name)

    #image_path = r"../frames/0.jpg"
    image_path = r"Untitled.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (28, 28))  # Resize to 28 by 28 pixels => ta distorcendo demais a imagem original
    normalized_img = resized_img / 255.0  # Normalize pixel values to [0, 1]

    # Flatten and reshape the image
    preprocessed_img = normalized_img.reshape(1, 28, 28)

    #predictions = model.predict(preprocessed_img)

    predictions = loaded_model.predict(preprocessed_img)

    # Display the original image and the predicted label
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"Predicted Number: {np.argmax(predictions)}")
    plt.show()