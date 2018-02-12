from __future__ import division, print_function
import os
import sys
import shutil
import warnings
import argparse
import numpy as np

from mlp_train_one import run_experiment

def experiment_loop(category_trials, exemplar_trials, params, results_path):
    stdout = sys.stdout
    # Create results_path folder. Remove previous one if it already exists.
    if os.path.isdir(results_path):
        warnings.warn('Removing old results folder of the same name!')
        shutil.rmtree(results_path)
    os.mkdir(results_path)
    np.save(os.path.join(results_path, 'category_trials.npy'),
            np.asarray(category_trials))
    np.save(os.path.join(results_path, 'exemplar_trials.npy'),
            np.asarray(exemplar_trials))
    # Loop through different values of (nb_categories, nb_exemplars)
    results_O1 = np.zeros(
        shape=(len(category_trials), len(exemplar_trials), params['nb_runs'])
    )
    results_O2 = np.zeros(
        shape=(len(category_trials), len(exemplar_trials), params['nb_runs'])
    )
    for i, nb_categories in enumerate(category_trials):
        for j, nb_exemplars in enumerate(exemplar_trials):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            log_file = os.path.join(results_path,
                                    'log_ca%0.4i_ex%0.4i' %
                                    (nb_categories, nb_exemplars))
            sys.stdout = open(log_file,'w')
            results_O1[i, j], results_O2[i, j] = \
                run_experiment(nb_categories, nb_exemplars, params)
            sys.stdout = stdout
            # Save results from this run to text file
            save_file_gen = os.path.join(results_path, 'results_O1.npy')
            np.save(save_file_gen, results_O1)
            save_file_gen = os.path.join(results_path, 'results_O2.npy')
            np.save(save_file_gen, results_O2)
    print('Experiment loop complete.')

def main():
    # Create the experiment parameter dictionary
    params = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'nb_runs': args.nb_runs,
        'nb_test': 1000
    }
    # Start the experiment loop
    category_trials = [2, 4, 8, 16, 32, 50]
    exemplar_trials = [3, 6, 9, 12, 15, 18]
    experiment_loop(
        category_trials=category_trials,
        exemplar_trials=exemplar_trials, params=params,
        results_path=args.save_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-r', '--nb_runs',
                        help='The number of training runs.',
                        required=False, type=int)
    parser.add_argument('-s', '--save_path',
                        help='The file path where results should be saved',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=200)
    parser.set_defaults(save_path='../results/mlp_results')
    parser.set_defaults(nb_runs=10)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()