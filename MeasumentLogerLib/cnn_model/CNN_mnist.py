# Source SVHN:
#   http://ufldl.stanford.edu/housenumbers/

# SVHN cropped digits Classification
# https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc#Street-View-House-Numbers-Classification

# https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import json

from keras.preprocessing import image
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

matplotlib.use('TkAgg')


def loss_acc_validation_graphs(model_history):
    acc = model_history['accuracy']
    val_acc = model_history['val_accuracy']
    loss = model_history['loss']
    val_loss = model_history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    '''
    =========================================================================================
    configs
    =========================================================================================
    '''
    rows = 28
    cols = 28
    img_layers = 1
    input_shape = (rows, cols, img_layers)

    batch_size = 128
    num_classes = 10
    epochs = 20

    train = False
    model_weights = r"model_mnist.h5"


    '''
    =========================================================================================
    Train data prep
    =========================================================================================
    '''
    if train or not os.path.exists(model_weights):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)

        '''
        =========================================================================================
        model layers configuration
        =========================================================================================
        '''
        model = tf.keras.models.Sequential()

        model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPool2D(2, 2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # separation validation data and fiting model
        x_val = x_train[50000:]
        x_train_1 = x_train[:50000]
        y_val = y_train[50000:]
        y_train_1 = y_train[:50000]

        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_val, y_val))

        history_dict = history.history
        with open('history.json', 'w') as json_file:
            json.dump(history_dict, json_file)

        model.save(model_weights) #o formato default de salvar modelos do tensorflow consiste em uma estrutura de arquivos, pesquisar como é a mesma depois
    else:
        print(f'Opening saved model: \'{model_weights}\'')

    '''
    =========================================================================================
    Loading model and making predition
    =========================================================================================
    '''
    loaded_model = tf.keras.models.load_model(model_weights)
    print(loaded_model.summary(), "\n\n")

    #ploting LOSS X ACC graph
    with open('history.json', 'r') as json_file:
        loaded_history = json.load(json_file)
    loss_acc_validation_graphs(loaded_history)


    #loading image
    image_path = r"6.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    inv_img = cv2.bitwise_not(img)
    #resized_img = cv2.resize(img, (28, 28))  # Resize to 28 by 28 pixels => ta distorcendo demais a imagem original
    normalized_img = inv_img / 255.0  # Normalize pixel values to [0 : 1]

    # Flatten and reshape the image
    preprocessed_img = normalized_img.reshape(1, 28, 28)
    predictions = loaded_model.predict(preprocessed_img)
    print("# prediction values: ", predictions)

    # Display the original image and the predicted label
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Number: {np.argmax(predictions)}")
    plt.show()