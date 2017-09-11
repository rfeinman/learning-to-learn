import argparse
import os
import pandas as pd

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import (synthesize_data, synthesize_new_data,
                                preprocess_data, evaluate_secondOrder)

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
    # Synthesize the training data.
    df, labels = synthesize_data(nb_categories,
                                 nb_exemplars,
                                 nb_textures,
                                 nb_colors)
    # Now we will create the test data set for the second-order generalization
    df_new, labels_new = synthesize_new_data(nb_categories,
                                             nb_textures,
                                             nb_colors)
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
    # Build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, batch_size=32, verbose=0)
    score = evaluate_secondOrder(model, X[test_inds], batch_size=32)

    return score

def save_results(cats, exemps, scores):
    df = pd.DataFrame()
    df['nb_categories'] = cats
    df['nb_exemplars'] = exemps
    df['score'] = scores
    df.to_csv('../results_secondOrder.csv', index=False)

def main(args):
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    # If a results file already exists, remove it.
    try:
        os.remove('../results_secondOrder.csv')
    except OSError:
        pass
    cats = []
    exemps = []
    scores = []
    # Loop through different values of (nb_categories, nb_exemplars)
    for nb_categories in range(50, 251, 50):
        for nb_exemplars in range(1, 8):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            result = run_experiment(nb_categories, nb_exemplars, 200, 200,
                                    args.nb_epochs)
            cats.append(nb_categories)
            exemps.append(nb_exemplars)
            scores.append(result)
            # Save results from this run to text file
            save_results(cats, exemps, scores)
    print('Experiment loop complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    args = parser.parse_args()
    main(args)