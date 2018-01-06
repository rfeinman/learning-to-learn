from __future__ import division, print_function
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.util import (evaluate_secondOrder, load_image_dataset,
                                 train_model)

def make_trial(shapes, colors, textures):
    # create a random trial
    ix = np.arange(len(shapes))
    while True:
        baseline = np.random.choice(ix)
        shape = shapes[baseline]
        color = colors[baseline]
        texture = textures[baseline]
        # only matches in shape
        shape_matches = \
        np.where((shapes == shape) &
                 (colors != color) &
                 (textures != texture))[0]
        # only matches in color
        color_matches = \
        np.where((shapes != shape) &
                 (colors == color) &
                 (textures != texture))[0]
        # only matches in texture
        texture_matches = \
        np.where((shapes != shape) &
                 (colors != color) &
                 (textures == texture))[0]

        if len(shape_matches) > 0 and len(color_matches) > 0 and len(
                texture_matches) > 0:
            break  # make sure we have an option for each image...

    shape_match = np.random.choice(shape_matches)
    color_match = np.random.choice(color_matches)
    texture_match = np.random.choice(texture_matches)

    return [baseline, shape_match, color_match, texture_match]

def build_test_trials(test_folder, nb_trials, target_size=(200, 200)):
    # First, load the images
    imgs = load_image_dataset(test_folder, target_size=target_size)
    # Collect the list of shapes, colors and textures
    feature_file = os.path.join(test_folder, 'data.csv')
    df = pd.read_csv(feature_file, index_col=0)
    shapes = df['shape'].as_matrix()
    colors = df['color'].as_matrix()
    textures = df['texture'].as_matrix()

    # Sample the trials
    ix = []
    for i in range(nb_trials):
        ix.extend(make_trial(shapes, colors, textures))
    ix = np.asarray(ix)
    return imgs[ix]

def run_experiment(nb_categories, nb_exemplars, params):
    # Set random seeds
    np.random.seed(0)
    tf.set_random_seed(0)
    # Create custom TF session if requested
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    data_folder = os.path.realpath('../data/images_generated')
    X, shapes = load_image_dataset(data_folder, nb_categories, nb_exemplars,
                                       target_size=params['img_size'])
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(shapes.reshape(-1, 1))
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(len(shapes))).difference(test_inds))
    test_folder = os.path.join(data_folder, 'test/')
    X_test = build_test_trials(test_folder, nb_trials=1000,
                               target_size=params['img_size'])
    # Build a neural network model and train it with the training set
    print('Training CNN model...')
    scores = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i + 1))
        model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
        # We're going to keep track of the best model throughout training,
        # monitoring the training loss
        weights_file = '../data/cnn_secondOrder.h5'
        if os.path.isfile(weights_file):
            os.remove(weights_file)
        checkpoint = ModelCheckpoint(
            weights_file,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        )
        train_model(
            model, X[train_inds], Y[train_inds], epochs=params['nb_epochs'],
            validation_data=None, batch_size=params['batch_size'],
            checkpoint=checkpoint
        )
        # Now that we've completed all training epochs, let's go ahead and
        # load the best model
        model.load_weights(weights_file)
        # Now evaluate the model on the test data
        score = evaluate_secondOrder(
            model, X_test, layer_num=-4,
            batch_size=params['batch_size']
        )
        scores.append(score)
    avg_score = np.mean(scores)
    print('\nScore: %0.4f' % avg_score)
    if params['gpu_options'] is not None:
        K.clear_session()
        sess.close()

    return avg_score

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
    acc = run_experiment(args.nb_categories, args.nb_exemplars, params)

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