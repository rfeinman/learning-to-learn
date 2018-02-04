from __future__ import division
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input
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

def simple_cnn_multiout(
        input_shape, nb_shapes, nb_colors, nb_textures, loss_weights=None
):
    layers = [
        # Conv, Pool
        Conv2D(5, (5, 5), padding='same', kernel_regularizer=l2(0.01)),
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
    ]
    image_in = Input(shape=input_shape, dtype='float32', name='image_in')
    x = layers[0](image_in)
    for layer in layers[1:]:
        x = layer(x)
    shape_out = Dense(nb_shapes, activation='softmax', name='shape_out')(x)
    color_out = Dense(nb_colors, activation='softmax', name='color_out')(x)
    texture_out = Dense(nb_textures, activation='softmax', name='texture_out')(x)

    model = Model(
        inputs=[image_in],
        outputs=[shape_out, color_out, texture_out]
    )
    if loss_weights is None:
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
        )
    else:
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            loss_weights=loss_weights
        )

    return model