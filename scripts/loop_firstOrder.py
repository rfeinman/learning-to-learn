import argparse
import numpy as np
import pandas as pd

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import synthesize_data, preprocess_data, save_results


def run_experiment(nb_categories, nb_exemplars):
    """
    TODO
    :param nb_categories:
    :param nb_exemplars:
    :return:
    """
    # Synthesize the training data. We will use one extra exemplar per category
    # as a test set - it will be separated later.
    df, labels = synthesize_data(nb_categories,
                                 nb_exemplars+1)
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
    model.fit(X[train_inds], Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
              verbose=0)
    loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0)

    return acc

def main():
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    cats = []
    exemps = []
    scores = []
    # Loop through different values of (nb_categories, nb_exemplars)
    for nb_categories in range(50, 251, 50):
        for nb_exemplars in range(1, 10):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            result = run_experiment(nb_categories, nb_exemplars)
            cats.append(nb_categories)
            exemps.append(nb_exemplars)
            scores.append(result)
            # Save results from this run to text file
            save_results(cats, exemps, scores, args.save_path)
    print('Experiment loop complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-sp', '--save_path',
                        help='The file path where results should be saved',
                        required=False, type=str)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(save_path='../results/results_firstOrder.csv')
    args = parser.parse_args()
    main()