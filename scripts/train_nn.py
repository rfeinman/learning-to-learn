import argparse
import os
import itertools
import warnings
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
#from keras.regularizers import l2


def generate_dictionary(column, nb_bits=None):
    """
    For now, do this randomly - but later make orthogonal
    """
    val_set = np.unique(column)
    nb_vals = val_set.shape[0]
    if nb_bits is None:
        # compute the required number of bits based on the
        # number of values. Let's add 2 bits for extra space.
        nb_bits = int(round(math.log(nb_vals, 2))) + 2
    else:
        assert type(nb_bits) == int
    # the candidate vectors
    candidates = np.asarray([seq for seq in itertools.product([0,1],
                                                              repeat=nb_bits)])
    # the candidates that we will use are randomly selected
    inds = np.random.choice(range(len(candidates)), nb_vals, replace=False)
    # now select the vectors
    vectors = candidates[inds]
    return {val_set[i]: vectors[i] for i in range(nb_vals)}

def convert_column(column, nb_bits=None):
    """
    Next, blah...
    """
    dictionary = generate_dictionary(column, nb_bits)
    return np.asarray([dictionary[elt] for elt in column])

def preprocess_data(df, labels, one_hot=False, nb_bits=None):
    """
    TODO
    :param df:
    :param labels:
    :param one_hot:
    :param nb_bits:
    :return:
    """
    if one_hot and nb_bits is not None:
        warnings.warn('nb_bits parameter is not used when one_hot=True.')
    if one_hot:
        # one-hot-encode the categorical variables in the dataset
        df = pd.get_dummies(df, columns=['shape', 'color', 'texture'])
        X = df.values
    else:
        # select bitmap vectors for each categorical variable
        shape = convert_column(df['shape'], nb_bits)
        color = convert_column(df['color'], nb_bits)
        texture = convert_column(df['texture'], nb_bits)
        X = np.concatenate((shape, color, texture), axis=1)
    # turn label words into number indices
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # one-hot-encode the labels so that they work with softmax output
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(y.reshape(-1, 1))

    return X, Y


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

def main(args):
    """
    TODO
    :param args:
    :return:
    """
    # load the data from text files
    df = pd.read_csv(os.path.realpath(args.data_path))
    labels = pd.read_csv(os.path.realpath(args.labels_path), header=None,
                         squeeze=True)
    # pre-process the data
    X, Y = preprocess_data(df, labels, one_hot=False, nb_bits=20)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)
    # split the data set into train-test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    # build a neural network model and train it with the training set
    model = build_NN(X.shape[-1], Y.shape[-1])
    model.fit(X_train, Y_train, epochs=args.nb_epochs, shuffle=True)
    loss, acc = model.evaluate(X_train, Y_train)
    print('\n Holdout accuracy: %0.02f%%' % (acc*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        help='The path to the data CSV file.',
                        required=True, type=str)
    parser.add_argument('-l', '--labels_path',
                        help='The path to the labels text file.',
                        required=True, type=str)
    parser.add_argument('-e', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=20)
    args = parser.parse_args()
    main(args)