from __future__ import division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.wrangle import (synthesize_data, get_train_test_parameters,
                                    build_train_set, build_test_trials_order1,
                                    build_test_set_order1)
from learning2learn.util import (train_model, train_test_split,
                                 evaluate_generalization)


def run_experiment(nb_categories, nb_exemplars, params):
    assert nb_categories <= 50
    # Set random seeds
    np.random.seed(0)
    # Create custom TF session if requested
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    # Get the shape, color, and texture parameters for the training and
    # testing sets. Features will be drawn from these parameter sets for each
    # sample
    (shape_set_train, shape_set_test), \
    (color_set_train, color_set_test), \
    (texture_set_train, texture_set_test) = \
        get_train_test_parameters(params['img_size'][0])
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
    # Build the training set
    print('Building the training set...')
    df_train, labels = synthesize_data(nb_categories, nb_exemplars)
    X_train = build_train_set(
        df_train, shape_set_train, color_set_train,
        texture_set_train, target_size=params['img_size']
    )
    ohe = OneHotEncoder(sparse=False)
    Y_train = ohe.fit_transform(labels.reshape(-1, 1))
    # Build the test set trials
    print('Building test trials...')
    X_test = build_test_trials_order1(
        df_train, shape_set_train, shape_set_test, color_set_train,
        color_set_test, texture_set_train, texture_set_test,
        nb_trials=2000, target_size=params['img_size']
    )
    X_test_acc, labels_test = build_test_set_order1(
        df_train, shape_set_train, color_set_test, texture_set_test,
        nb_trials=2000, target_size=params['img_size']
    )
    Y_test = ohe.transform(labels_test.reshape(-1, 1))
    scores = []
    accs = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i+1))
        # Build a neural network model and train it with the training set
        model = simple_cnn(
            input_shape=X_train.shape[1:],
            nb_classes=Y_train.shape[-1]
        )
        # We're going to keep track of the best model throughout training,
        # monitoring the training loss
        weights_file = '../data/cnn_firstOrder.h5'
        if os.path.isfile(weights_file):
            os.remove(weights_file)
        checkpoint = ModelCheckpoint(
            weights_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            period=2
        )
        # We'll provide the test set as 'validation data' merely so we can
        # monitor the trajectory... the network won't be using this data.
        train_model(
            model, X_train, Y_train, epochs=params['nb_epochs'],
            validation_data=None, batch_size=params['batch_size'],
            checkpoint=checkpoint
        )
        # Now that we've completed all training epochs, let's go ahead and
        # load the best model
        model.load_weights(weights_file)
        # Now evaluate the model on the test data
        score = evaluate_generalization(
            model, X_test, layer_num=-4,
            batch_size=128
        )
        _, acc = model.evaluate(
            X_test_acc, Y_test, verbose=0,
            batch_size=params['batch_size']
        )
        scores.append(score)
        accs.append(acc)
        print('\nScore: %0.4f' % score)
        print('\nAccuracy: %0.4f' % acc)
    if params['gpu_options'] is not None:
        K.clear_session()
        sess.close()

    return np.asarray(scores), np.asarray(accs)

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
        'nb_trials': 1,
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
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(nb_categories=10)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()