import argparse
import numpy as np

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import synthesize_data, preprocess_data


def run_experiment(nb_categories, nb_exemplars, nb_textures, nb_colors,
                   nb_epochs):
    # synthesize the training data. we will use one extra exemplar per category
    # as a test set - it will be separated later.
    df, labels = synthesize_data(nb_categories,
                                 nb_exemplars+1,
                                 nb_textures,
                                 nb_colors)
    # pre-process the data
    X, Y = preprocess_data(df, labels, one_hot=False, nb_bits=20)
    # now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    train_inds = list(set(range(df.shape[0])).difference(test_inds))
    # check to make sure we did this right
    for i in range(nb_categories):
        assert np.array_equal(X[test_inds][i][:20],
                              X[train_inds][i * nb_exemplars][:20])
    # build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
              verbose=0)
    loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0)
    return acc

def print_results(results):
    with open('../results.txt', 'w') as f:
        for elt in results:
            f.write('%s: %s\n' % (elt, results[elt]))

def main(args):
    out_file = open('../results.txt', 'w')
    results = {}
    #for nb_categories in [100, 500, 1000, 5000, 10000]:
    #    for nb_exemplars in [3, 5, 10, 20]:
    for nb_categories in [100, 500, 1000, 5000]:
        for nb_exemplars in [3, 5, 10]:
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            key = 'cat_%i_ex_%i' % (nb_categories, nb_exemplars)
            results[key] = run_experiment(nb_categories, nb_exemplars, 200,
                                          200, 100)
            #print_results(results)
            out_file.write('cat %0.6i, ex %0.2i: %0.3f\n' %
                           (nb_categories, nb_exemplars, results[key]))
    out_file.close()
    print('Experiment loop complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=20)
    args = parser.parse_args()
    main(args)