from __future__ import division, print_function
import argparse
import numpy as np
from keras.callbacks import EarlyStopping

from learning2learn.models import simple_mlp
from learning2learn.wrangle import synthesize_data, preprocess_data
from learning2learn.util import add_noise

def run_experiment(nb_categories, nb_exemplars, params):
    """
    TODO
    :param nb_categories:
    :param nb_exemplars:
    :return:
    """
    # Synthesize the training data. We will use one extra exemplar per category
    # as a test set - it will be separated later.
    df, labels = synthesize_data(nb_categories, nb_exemplars+1)
    # Pre-process the data
    X, Y = preprocess_data(df, labels, one_hot=False, nb_bits=20)
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    train_inds = list(set(range(df.shape[0])).difference(test_inds))
    # Add noise if indicated by command line params
    if params['noise'] > 0:
        print('Adding noise with p=%0.2f' % params['noise'])
        X_train = add_noise(X[train_inds], p=params['noise'])
    else:
        X_train = X[train_inds]
    # Build a neural network model and train it with the training set
    scores = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i + 1))
        model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
        model.fit(X_train, Y[train_inds], epochs=params['nb_epochs'],
                  shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
                  verbose=1, batch_size=params['batch_size'],
                  callbacks=[EarlyStopping(monitor='loss', patience=5)])
        loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0,
                                   batch_size=params['batch_size'])
        scores.append(acc)

    return np.mean(scores)

def main():
    params = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'noise': args.noise,
        'nb_trials': 1
    }
    acc = run_experiment(args.nb_categories, args.nb_exemplars, params)
    print('\nAccuracy: %0.4f' % score)

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
    parser.set_defaults(nb_epochs=200)
    parser.set_defaults(nb_categories=100)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(noise=0.)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    assert args.noise >= 0. and args.noise <= 1., "'noise' parameter must be " \
                                                  "a float between 0-1."
    main()