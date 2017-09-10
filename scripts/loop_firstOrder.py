import argparse
import os
import numpy as np

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import synthesize_data, preprocess_data


def run_experiment(nb_categories, nb_exemplars, nb_textures, nb_colors,
                   nb_epochs):
    """
    TODO
    :param nb_categories:
    :param nb_exemplars:
    :param nb_textures:
    :param nb_colors:
    :param nb_epochs:
    :return:
    """
    # Synthesize the training data. We will use one extra exemplar per category
    # as a test set - it will be separated later.
    df, labels = synthesize_data(nb_categories,
                                 nb_exemplars+1,
                                 nb_textures,
                                 nb_colors)
    # Pre-process the data
    X, Y = preprocess_data(df, labels, one_hot=False, nb_bits=20)
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    train_inds = list(set(range(df.shape[0])).difference(test_inds))
    # Check to make sure we did this right
    for i in range(nb_categories):
        assert np.array_equal(X[test_inds][i][:20],
                              X[train_inds][i * nb_exemplars][:20])
    # Build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
              verbose=0)
    loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0)

    return acc

def main(args):
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    # If a results file already exists, remove it.
    try:
        os.remove('../results_firstOrder.txt')
    except OSError:
        pass
    results = {}
    # Loop through different values of (nb_categories, nb_exemplars)
    for nb_categories in [100, 500, 1000, 5000]:
        for nb_exemplars in [3, 5, 10]:
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            key = 'cat_%i_ex_%i' % (nb_categories, nb_exemplars)
            results[key] = run_experiment(nb_categories, nb_exemplars, 200,
                                          200, args.nb_epoch)
            # Save results from this run to text file
            with open('../results_firstOrder.txt', 'a') as f:
                f.write('cat %0.6i, ex %0.2i: %0.3f\n' %
                        (nb_categories, nb_exemplars, results[key]))
    print('Experiment loop complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    args = parser.parse_args()
    main(args)