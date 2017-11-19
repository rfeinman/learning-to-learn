from __future__ import division, print_function
import argparse

from learning2learn.util import experiment_loop
from mlp_train_firstOrder import run_experiment

def main():
    params = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'noise': args.noise,
        'nb_trials': 5
    }
    category_trials = range(5, 51, 5)
    exemplar_trials = range(1, 15)
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
    parser.add_argument('-no', '--noise',
                        help='Noise fraction; binomial probability between '
                             '0-1.',
                        required=False, type=float)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(noise=0.)
    parser.set_defaults(save_path='../results/mlp_results_firstOrder')
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    assert args.noise >= 0. and args.noise <= 1., "'noise' parameter must be " \
                                                  "a float between 0-1."
    main()