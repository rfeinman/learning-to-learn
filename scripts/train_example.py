"""
Train a simple neural network using a pre-specified data set.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import preprocess_data

def main(args):
    """
    The main script code.
    :param args: (Namespace object) command line arguments
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
    model = simple_mlp(X.shape[-1], Y.shape[-1])
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