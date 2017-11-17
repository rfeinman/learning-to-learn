from __future__ import division
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.images import generate_dataset_parameters, generate_image
from learning2learn.util import (synthesize_data, save_results,
                                 load_image_dataset)

def create_dataset(nb_categories, nb_exemplars, data_folder):
    # Generate the set of shapes, colors and textures that we will draw from
    shape_set, color_set, texture_set = \
        generate_dataset_parameters(nb_categories)
    # Create folder where images will be stored; if exists, remove
    if os.path.isdir(data_folder):
        print('A dataset of the specified parameters already exists. Using '
              'the existing one...')
    else:
        print('Building the image dataset...')
        os.mkdir(data_folder)
        # Synthesize the dataset. Use nb_exemplars+1 because 1 exemplar
        # of each class will be used for testing.
        df, _ = synthesize_data(nb_categories, nb_exemplars+1)
        shapes = [shape_set[i] for i in df['shape']]
        colors = [color_set[i] for i in df['color']]
        textures = [texture_set[i] for i in df['texture']]
        for shape, color, texture, i in \
                zip(shapes, colors, textures, range(len(shapes))):
            save_file = os.path.join(data_folder, 'img%0.4i.png' % i)
            generate_image(shape, color, texture, save_file)
        # Save the dataset parameters so we know what we're working with
        df.to_csv(os.path.join(data_folder, 'data.csv'))

def run_experiment(nb_categories, nb_exemplars, gpu_options=None):

    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    if gpu_options is not None:
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
    data_folder = os.path.realpath('../data/images_ca%0.4i_ex%0.4i' %
                                   (nb_categories, nb_exemplars))
    create_dataset(nb_categories, nb_exemplars, data_folder)
    X, shapes = load_image_dataset(data_folder, target_size=(200, 200))
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(shapes.reshape(-1, 1))
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(len(shapes))).difference(test_inds))
    # Build a neural network model and train it with the training set
    print('Training CNN model...')
    model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
              verbose=0, batch_size=args.batch_size)
    loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0,
                               batch_size=args.batch_size)
    if gpu_options is not None:
        K.clear_session()
        sess.close()

    return acc

def main(gpu_options=None):
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    cats = []
    exemps = []
    scores = []
    # Loop through different values of (nb_categories, nb_exemplars)
    for nb_categories in range(5, 51, 5):
        for nb_exemplars in range(1, 15):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            result = run_experiment(nb_categories, nb_exemplars, gpu_options)
            cats.append(nb_categories)
            exemps.append(nb_exemplars)
            scores.append(result)
            # Save results from this run to text file
            save_results(cats, exemps, scores, args.save_path)
    print('Experiment loop complete.')

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
    parser.set_defaults(save_path='../results/cnn_results_firstOrder.csv')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    tf.set_random_seed(0)
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        main(gpu_options)
    else:
        main()