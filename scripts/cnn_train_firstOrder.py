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
from learning2learn.old import get_train_test_inds
from learning2learn.wrangle import (synthesize_data, get_train_test_parameters,
                                    build_train_set)
from learning2learn.util import train_model, subsample


def run_experiment(nb_categories, nb_exemplars, params):
    assert nb_categories+params['nb_test'] <= 50
    # Set random seeds
    np.random.seed(0)
    # Create custom TF session if requested
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    # First, get the parameters for the training and testing sets. This step
    # is independent of the input dataset df_train. The same breakdown is used
    # each time.
    df, labels = synthesize_data(nb_categories, nb_exemplars+params['nb_test'])
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(labels.reshape(-1, 1))
    (shape_set_train, shape_set_test), \
    (color_set_train, color_set_test), \
    (texture_set_train, texture_set_test) = \
        get_train_test_parameters(params['img_size'][0])
    print('Training CNN model...')
    scores = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i+1))
        # Build the training set of images. Do this inside the loop because
        # a slightly random subset of the parameters is selected each time;
        # helps get some variance
        if nb_categories < 50:
            shape_set_train = subsample(shape_set_train, nb_categories)
            color_set_train = subsample(color_set_train, nb_categories)
            texture_set_train = subsample(texture_set_train, nb_categories)
        print('Building the training set...')
        X = build_train_set(df, shape_set_train, color_set_train,
                            texture_set_train, target_size=params['img_size'])
        # Now, we separate the train and test sets
        train_inds, test_inds = get_train_test_inds(nb_categories, nb_exemplars,
                                                    len(labels),
                                                    params['nb_test'])
        # Build a neural network model and train it with the training set
        model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
        # We're going to keep track of the best model throughout training,
        # monitoring the training loss
        weights_file = '../data/cnn_firstOrder.h5'
        if os.path.isfile(weights_file):
            os.remove(weights_file)
        checkpoint = ModelCheckpoint(
            weights_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        )
        # We'll provide the test set as 'validation data' merely so we can
        # monitor the trajectory... the network won't be using this data.
        train_model(
            model, X[train_inds], Y[train_inds], epochs=params['nb_epochs'],
            validation_data=None, batch_size=params['batch_size'],
            checkpoint=checkpoint
        )
        # Now that we've completed all training epochs, let's go ahead and
        # load the best model
        model.load_weights(weights_file)
        # Now evaluate the model on the test data
        loss, acc = model.evaluate(
            X[test_inds], Y[test_inds], verbose=0,
            batch_size=128
        )
        scores.append(acc)
        print('\nScore: %0.4f' % acc)
    if params['gpu_options'] is not None:
        K.clear_session()
        sess.close()

    return np.asarray(scores)

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
        'nb_trials': 1,
        'img_size': (200, 200),
        'gpu_options': gpu_options
    }
    # Run the experiment
    scores = run_experiment(args.nb_categories, args.nb_exemplars, params)

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