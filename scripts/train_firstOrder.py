from __future__ import division
import argparse
import numpy as np

from learning2learn.models import simple_mlp
from learning2learn.util import synthesize_data, preprocess_data, add_noise


def main():
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    # Synthesize the training data. we will use one extra exemplar per category
    # as a test set - it will be separated later.
    df, labels = synthesize_data(args.nb_categories,
                                 args.nb_exemplars+1)
    # Pre-process the data
    X, Y = preprocess_data(df, labels, one_hot=False, nb_bits=20)
    # Add noise if indicated by command line params
    if args.noise > 0.:
        X = add_noise(X, p=args.noise)
    # Now, we separate the train and test sets
    test_inds = [i*(args.nb_exemplars+1) for i in range(args.nb_categories)]
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(df.shape[0])).difference(test_inds))
    # Add noise if indicated by command line params
    if args.noise > 0.:
        print('Adding noise with p=%0.2f' % args.noise)
        X_train = add_noise(X[train_inds], p=args.noise)
    else:
        X_train = X[train_inds]
    # Build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X_train, Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]))

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
    parser.set_defaults(nb_epochs=200)
    parser.set_defaults(nb_categories=100)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(noise=0.)
    args = parser.parse_args()
    assert args.noise >= 0. and args.noise <= 1., "'noise' parameter must be " \
                                                  "a float between 0-1."
    main()