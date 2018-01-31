from __future__ import division
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.regularizers import l2

def build_model(layers):
    """
    Takes in a list of Keras layers and returns an initialized
    Keras Sequential model with those layers.
    :param layers: (list) the list of Keras layers for the model
    :return: (keras Sequential) a compiled Keras model
    """
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model

def simple_mlp(nb_in, nb_classes):
    """
    Construct a basic one-layer neural net using Keras Sequential.
    :param nb_in: (int) the size of the input vector
    :param nb_classes: (int) the number of classes for the model
    :return: (keras Sequential) a compiled Keras model
    """
    layers = [
        Dense(30, input_dim=nb_in, kernel_regularizer=l2(0.01)),
        #Dense(30, input_dim=nb_in),
        Activation('relu'),
        Dense(nb_classes),
        Activation('softmax')
    ]

    return build_model(layers)

def simple_cnn(input_shape, nb_classes):
    layers = [
        # Conv, Pool
        Conv2D(5, (5, 5), padding='same', input_shape=input_shape,
               kernel_regularizer=l2(0.01)),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        # Conv, Pool
        Conv2D(5, (5, 5), padding='same', kernel_regularizer=l2(0.01)),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        # Flatten
        Flatten(),
        # Hidden layer
        Dropout(0.2),
        Dense(25, kernel_regularizer=l2(0.01)),
        Activation('relu'),
        # Output layer
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ]

    return build_model(layers)

def simple_cnn_old1(input_shape, nb_classes):
    layers = [
        # Conv, Pool
        Conv2D(5, (5, 5), padding='same', input_shape=input_shape,
               kernel_regularizer=l2(0.01)),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        # Conv, Pool
        Conv2D(5, (5, 5), padding='same', kernel_regularizer=l2(0.01)),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        # Flatten
        Flatten(),
        # Hidden layer
        Dense(25, kernel_regularizer=l2(0.01)),
        Activation('relu'),
        # Output layer
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ]

    return build_model(layers)

def simple_cnn_old2(input_shape, nb_classes):
    layers = [
        # Conv, Pool
        Conv2D(32, (5, 5), padding='same', input_shape=input_shape),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        # Conv, Pool
        Conv2D(64, (5, 5), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(5, 5)),
        # Flatten
        Flatten(),
        # Hidden layer
        Dropout(0.5),
        Dense(2048),
        Activation('relu'),
        # Output layer
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ]

    return build_model(layers)