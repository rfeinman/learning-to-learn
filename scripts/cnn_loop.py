from __future__ import division, print_function
import argparse
import tensorflow as tf

from learning2learn.util import experiment_loop
from cnn_train_one import run_experiment

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
        'nb_runs': args.nb_runs,
        'nb_test': 1000,
        'img_size': (200, 200),
        'gpu_options': gpu_options
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
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=400)
    parser.set_defaults(nb_runs=10)
    parser.set_defaults(save_path='../results/cnn_results_combined')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()