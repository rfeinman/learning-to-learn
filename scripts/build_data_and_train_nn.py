import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import synthesize_data, preprocess_data

def main(args):
    """
    TODO
    :param args:
    :return:
    """
    # load the data from text files
    df, labels = synthesize_data(args.nb_categories, args.nb_exemplars,
                                 args.nb_textures, args.nb_colors)
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
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=True, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=True, type=int)
    parser.add_argument('-t', '--nb_textures',
                        help='The number of textures.',
                        required=True, type=int)
    parser.add_argument('-co', '--nb_colors',
                        help='The number of colors.',
                        required=True, type=int)
    parser.set_defaults(nb_epochs=20)
    args = parser.parse_args()
    main(args)