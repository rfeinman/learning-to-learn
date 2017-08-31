import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.regularizers import l2


def preprocess_data(df, labels):
    """
    TODO
    :param df:
    :param labels:
    :return:
    """
    # one-hot-encode the categorical variables in the dataset
    # TODO: update this; should be bitmaps rather than OHE vectors
    df = pd.get_dummies(df, columns=['shape', 'color', 'texture'])
    # turn label words into number indices
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # one-hot-encode the labels so that they work with softmax output
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(y.reshape(-1, 1))

    return df.values, Y


def build_NN(nb_in, nb_classes):
    """
    Construct a basic one-layer neural net using keras Sequential.
    :param nb_in: (int) the size of the input vector
    :param nb_classes: (int) the number of classes for the model
    :return: (keras Sequential) a compiled keras model
    """
    model = Sequential()
    model.add(Dense(30, input_dim=nb_in, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes, kernel_regularizer=l2(0.01)))
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
    X, Y = preprocess_data(df, labels)
    # split the data set into train-test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    # build a neural network model and train it with the training set
    model = build_NN(X.shape[-1], Y.shape[-1])
    model.fit(X_train, Y_train, epochs=args.nb_epochs)
    loss, acc = model.evaluate(X_train, Y_train)
    print('Accuracy: %0.02f%%' % (acc*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        help='The path to the data CSV file.',
                        required=False, type=str)
    parser.add_argument('-l', '--labels_path',
                        help='The path to the labels text file.',
                        required=False, type=str)
    parser.add_argument('-e', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.set_defaults(data_path='../data/smith_objects.csv')
    parser.set_defaults(labels_path='../data/smith_labels.csv')
    parser.set_defaults(nb_epochs=20)
    args = parser.parse_args()
    main(args)