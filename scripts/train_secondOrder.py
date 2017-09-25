import argparse
import pandas as pd

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import (synthesize_data, synthesize_new_data,
                                preprocess_data, evaluate_secondOrder,
                                add_noise)

def main():
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    # Synthesize the training data.
    df, labels = synthesize_data(args.nb_categories,
                                 args.nb_exemplars)
    # Now we will create the test data set for the second-order generalization
    df_new, labels_new = synthesize_new_data(args.nb_categories)
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
    if args.noise > 0.:
        print('Adding noise with p=%0.2f' % args.noise)
        X_train = add_noise(X[train_inds], p=args.noise)
    else:
        X_train = X[train_inds]
    # Build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X_train, Y[train_inds], epochs=args.nb_epochs, shuffle=True)
    score = evaluate_secondOrder(model, X[test_inds], batch_size=32)
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
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(nb_categories=100)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(noise=0.)
    args = parser.parse_args()
    assert args.noise >= 0. and args.noise <= 1., "'noise' parameter must be " \
                                                  "a float between 0-1."
    main()