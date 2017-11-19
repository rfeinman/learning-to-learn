from __future__ import division
import argparse
import numpy as np
import pandas as pd

from learning2learn.models import simple_mlp
from learning2learn.util import (synthesize_data, synthesize_new_data,
                                 preprocess_data, evaluate_secondOrder,
                                 add_noise)

def run_experiment(nb_categories, nb_exemplars, params):
    """
    TODO
    :param nb_categories:
    :param nb_exemplars:
    :return:
    """
    # Synthesize the training data.
    df, labels = synthesize_data(nb_categories, nb_exemplars)
    # Now we will create the test data set for the second-order generalization
    df_new, labels_new = synthesize_new_data(nb_categories)
    # Check to make sure that all new feature values have not been seen in
    # training
    for feature in ['shape', 'color', 'texture']:
        intersection = set(df_new[feature]).intersection(set(df[feature]))
        assert len(intersection) == 0
    # Concatenate and pre-process the data
    X, Y = preprocess_data(pd.concat([df, df_new]),
                           pd.concat([labels, labels_new]),
                           one_hot=False, nb_bits=20)
    # Keep track of location of train and test sets
    train_inds = range(df.shape[0])
    test_inds = range(df.shape[0], X.shape[0])
    # Add noise if indicated by command line params
    if params['noise'] > 0:
        print('Adding noise with p=%0.2f' % params['noise'])
        X_train = add_noise(X[train_inds], p=params['noise'])
    else:
        X_train = X[train_inds]
    # Build a neural network model and train it with the training set
    scores = []
    for _ in range(params['nb_trials']):
        model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
        model.fit(X_train, Y[train_inds], epochs=params['nb_epochs'],
                  shuffle=True, verbose=1, batch_size=params['batch_size'])
        score = evaluate_secondOrder(model, X[test_inds], layer_num=1,
                                     batch_size=params['batch_size'])
        scores.append(score)

    return np.mean(scores)

def main():
    params = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'noise': args.noise,
        'nb_trials': 1
    }
    score = run_experiment(args.nb_categories, args.nb_exemplars, params)
    print('\nScore: %0.4f' % score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-no', '--noise',
                        help='Noise fraction; binomial probability between '
                             '0-1.',
                        required=False, type=float)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(nb_categories=100)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(noise=0.)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    assert args.noise >= 0. and args.noise <= 1., "'noise' parameter must be " \
                                                  "a float between 0-1."
    main()