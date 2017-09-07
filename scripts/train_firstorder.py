import argparse
import numpy as np

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import synthesize_data, preprocess_data


def main(args):
    """
    The main script code.
    :param args: (Namespace object) command line arguments
    """
    # synthesize the training data. we will use one extra exemplar per category
    # as a test set - it will be separated later.
    df, labels = synthesize_data(args.nb_categories,
                                 args.nb_exemplars+1,
                                 args.nb_textures,
                                 args.nb_colors)
    # pre-process the data
    X, Y = preprocess_data(df, labels, one_hot=False, nb_bits=20)
    # now, we separate the train and test sets
    test_inds = [i*(args.nb_exemplars+1) for i in range(args.nb_categories)]
    train_inds = list(set(range(df.shape[0])).difference(test_inds))
    # check to make sure we did this right
    for i in range(args.nb_categories):
        assert np.array_equal(X[test_inds][i][:20],
                              X[train_inds][i*args.nb_exemplars][:20])
    print('X_train shape: ', X[train_inds].shape)
    print('Y_train shape: ', Y[train_inds].shape)
    print('X_test shape: ', X[test_inds].shape)
    print('Y_test shape: ', Y[test_inds].shape)
    # build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]))
    loss, acc = model.evaluate(X[test_inds], Y[test_inds])
    #print('\nHoldout accuracy: %0.04f' % acc)

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
    parser.add_argument('-te', '--nb_textures',
                        help='The number of textures.',
                        required=True, type=int)
    parser.add_argument('-co', '--nb_colors',
                        help='The number of colors.',
                        required=True, type=int)
    parser.set_defaults(nb_epochs=20)
    args = parser.parse_args()
    main(args)