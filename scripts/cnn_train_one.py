from __future__ import division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder

from learning2learn.models import simple_cnn
from learning2learn.wrangle import synthesize_data
from learning2learn.wrangle import get_train_test_parameters
from learning2learn.wrangle import build_train_set
from learning2learn.wrangle import build_test_trials_O1
from learning2learn.wrangle import build_test_trials_O2
from learning2learn.util import train_model
from learning2learn.util import train_test_split
from learning2learn.util import evaluate_generalization


def run_experiment(nb_categories, nb_exemplars, params):
    assert nb_categories <= 50
    # enforce a maximum batch size to avoid doing full-batch GD
    batch_size = min(
        params['batch_size'],
        int(np.floor(nb_categories * nb_exemplars / 5))
    )
    # Create custom TF session if requested
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    # Set random seed
    np.random.seed(0)
    # Allocate space for storing results
    scores_O1 = np.zeros(params['nb_runs'])
    scores_O2 = np.zeros(params['nb_runs'])
    for i in range(params['nb_runs']):
        print('Round #%i' % (i+1))
        # Build the training set
        print('Building the training set...')
        df_train = synthesize_data(nb_categories, nb_exemplars)
        labels = df_train['shape'].values
        ohe = OneHotEncoder(sparse=False)
        Y_train = ohe.fit_transform(labels.reshape(-1, 1))
        # Get the shape, color, and texture parameters for the training and
        # testing sets. Features will be drawn from these parameter sets for
        # each sample
        (shape_set_train, shape_set_test), \
        (color_set_train, color_set_test), \
        (texture_set_train, texture_set_test) = \
            get_train_test_parameters(images=True,
                                      img_size=params['img_size'][0])
        if nb_categories < 50:
            shape_set_train, _ = train_test_split(
                shape_set_train,
                50 - nb_categories
            )
            color_set_train, _ = train_test_split(
                color_set_train,
                50 - nb_categories
            )
            texture_set_train, _ = train_test_split(
                texture_set_train,
                50 - nb_categories
            )
        X_train = build_train_set(
            df_train, shape_set_train, color_set_train,
            texture_set_train, target_size=params['img_size'],
            shift_scale=20
        )
        # Build the 1st-order test set trials
        print('Building test trials...')
        X_test_O1 = build_test_trials_O1(
            df_train, shape_set_train, shape_set_test, color_set_train,
            color_set_test, texture_set_train, texture_set_test,
            nb_trials=params['nb_test'], target_size=params['img_size'],
            shift_scale=20
        )
        # Build the 2nd-order test set trials
        X_test_O2 = build_test_trials_O2(
            shape_set_test, color_set_test, texture_set_test,
            nb_trials=params['nb_test'], target_size=params['img_size'],
            shift_scale=20
        )
        # Build a neural network model and train it with the training set
        model = simple_cnn(
            input_shape=X_train.shape[1:],
            nb_classes=Y_train.shape[-1]
        )
        # We're going to keep track of the best model throughout training,
        # monitoring the training loss
        weights_file = '../data/cnn_combined.h5'
        if os.path.isfile(weights_file):
            os.remove(weights_file)
        checkpoint = ModelCheckpoint(
            weights_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            period=2
        )
        train_model(
            model, X_train, Y_train, epochs=params['nb_epochs'],
            validation_data=None, batch_size=batch_size,
            checkpoint=checkpoint, burn_period=50
        )
        # Now that we've completed all training epochs, let's go ahead and
        # load the best model
        model.load_weights(weights_file)
        # Now evaluate the model on the test data
        scores_O1[i] = evaluate_generalization(
            model, X_test_O1, layer_num=-4,
            batch_size=128
        )
        scores_O2[i] = evaluate_generalization(
            model, X_test_O2, layer_num=-4,
            batch_size=128
        )
    # Close custom TF session if we started one
    if params['gpu_options'] is not None:
        K.clear_session()
        sess.close()
    print('\n1st-order generalization score: %0.4f' % scores_O1.mean())
    print('\n2nd-order generalization score: %0.4f' % scores_O2.mean())

    return scores_O1, scores_O2

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
    # Run the experiment
    _ = run_experiment(args.nb_categories, args.nb_exemplars, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-c', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-e', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-r', '--nb_runs',
                        help='The number of training runs.',
                        required=False, type=int)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=400)
    parser.set_defaults(nb_categories=10)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(nb_runs=10)
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()