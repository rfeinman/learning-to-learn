from keras.models import Sequential
from keras.layers.core import Dense, Activation
#from keras.regularizers import l2

def build_NN(nb_in, nb_classes):
    """
    Construct a basic one-layer neural net using keras Sequential.
    :param nb_in: (int) the size of the input vector
    :param nb_classes: (int) the number of classes for the model
    :return: (keras Sequential) a compiled keras model
    """
    model = Sequential()
    #model.add(Dense(30, input_dim=nb_in, kernel_regularizer=l2(0.01)))
    model.add(Dense(30, input_dim=nb_in))
    model.add(Activation('relu'))
    #model.add(Dense(nb_classes, kernel_regularizer=l2(0.01)))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model