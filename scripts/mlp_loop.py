from __future__ import division, print_function
import argparse

from learning2learn.util import experiment_loop
from mlp_train_one import run_experiment


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
        run_experiment, category_trials=category_trials,
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
    parser.set_defaults(save_path='../results/mlp_results_combined')
    parser.set_defaults(nb_runs=10)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()