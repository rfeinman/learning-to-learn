from __future__ import division, print_function

import os
import sys
import warnings
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')

#from learning2learn.util import experiment_loop
from cnn_train_firstOrder import run_experiment

def experiment_loop(exectue_fn, category_trials, exemplar_trials, params,
                    results_path):
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
    results_gen = np.zeros(
        shape=(len(category_trials), len(exemplar_trials), params['nb_trials'])
    )
    results_acc = np.zeros(
        shape=(len(category_trials), len(exemplar_trials), params['nb_trials'])
    )
    for i, nb_categories in enumerate(category_trials):
        for j, nb_exemplars in enumerate(exemplar_trials):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            log_file = os.path.join(results_path,
                                    'log_ca%0.4i_ex%0.4i' %
                                    (nb_categories, nb_exemplars))
            sys.stdout = open(log_file,'w')
            results_gen[i, j], results_acc[i, j] = exectue_fn(
                nb_categories, nb_exemplars, params
            )
            sys.stdout = stdout
            # Save results from this run to text file
            save_file_gen = os.path.join(results_path, 'results_gen.npy')
            np.save(save_file_gen, results_gen)
            save_file_acc = os.path.join(results_path, 'results_acc.npy')
            np.save(save_file_acc, results_acc)
    print('Experiment loop complete.')

def main():
    # GPU settings
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
    else:
        gpu_options = None
    # Create the experiment parameter dictionary
    params = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'nb_test': 5,
        'nb_trials': 5,
        'img_size': (200, 200),
        'gpu_options': gpu_options
    }
    # Start the experiment loop
    category_trials = [5, 8, 12, 15, 18, 21]
    exemplar_trials = [10, 20, 30, 40, 50]
    #category_trials = np.arange(3, 21)
    #exemplar_trials = np.arange(1, 10)
    experiment_loop(
        run_experiment, category_trials=category_trials,
        exemplar_trials=exemplar_trials, params=params,
        results_path=args.save_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-sp', '--save_path',
                        help='The file path where results should be saved',
                        required=False, type=str)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(save_path='../results/cnn_results_firstOrder')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()