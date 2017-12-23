from __future__ import division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.images import generate_dataset_parameters, generate_image
from learning2learn.util import (synthesize_data, synthesize_new_data,
                                 evaluate_secondOrder, load_image_dataset)

def create_dataset(nb_categories, nb_exemplars, data_folder):
    # Generate the set of shapes, colors and textures that we will draw from
    shape_set, color_set, texture_set = \
        generate_dataset_parameters(nb_categories)
    # Create folder where images will be stored; if exists, remove
    if os.path.isdir(data_folder):
        print('A dataset of the specified parameters already exists. Using '
              'the existing one...')
    else:
        print('Building the training dataset...')
        os.mkdir(data_folder)
        # Synthesize the dataset.
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
    f = os.path.join(data_folder, 'test0000/')
    if os.path.isdir(f):
        print('The test set already exists. Using existing one...')
    else:
        print('The test sets do not exist. Building the test sets...')
        # Now Synthesize the test datasets
        df_test, _ = synthesize_new_data(nb_categories)
        shape_set_test, color_set_test, texture_set_test = \
            generate_dataset_parameters(3*nb_categories)
        shapes_test = [shape_set_test[i-nb_categories]
                       for i in df_test['shape']]
        colors_test = [color_set_test[i-nb_categories]
                       for i in df_test['color']]
        textures_test = [texture_set_test[i-nb_categories]
                         for i in df_test['texture']]
        for i in range(int(len(df_test)/4)):
            test_folder = os.path.join(data_folder, 'test%0.4i' % i)
            os.mkdir(test_folder)
            img_file = os.path.join(test_folder, '0-base.png')
            generate_image(shapes_test[i*4], colors_test[i*4],
                           textures_test[i*4], img_file)
            img_file = os.path.join(test_folder, '1-shape_match.png')
            generate_image(shapes_test[i*4+1], colors_test[i*4+1],
                           textures_test[i*4+1], img_file)
            img_file = os.path.join(test_folder, '2-color_match.png')
            generate_image(shapes_test[i*4+2], colors_test[i*4+2],
                           textures_test[i*4+2], img_file)
            img_file = os.path.join(test_folder, '3-texture_match.png')
            generate_image(shapes_test[i*4+3], colors_test[i*4+3],
                           textures_test[i*4+3], img_file)

def get_test_set(data_folder, target_size):
    contents = [elt for elt in os.listdir(data_folder)
                if elt.startswith('test')]
    contents = sorted(contents)
    X = []
    for dir in contents:
        dir_path = os.path.join(data_folder, dir)
        imgs = load_image_dataset(dir_path, target_size=target_size,
                                  feature_info=False)
        X.append(imgs)
    X = np.concatenate(X)

    return X

def run_experiment(nb_categories, nb_exemplars, params):
    if params['gpu_options'] is not None:
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=params['gpu_options'])
        )
        K.set_session(sess)
    data_folder = os.path.realpath('../data/images_ca%0.4i_ex%0.4i' %
                                   (nb_categories, nb_exemplars))
    create_dataset(nb_categories, nb_exemplars, data_folder)
    X, shapes = load_image_dataset(data_folder, target_size=params['img_size'])
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(shapes.reshape(-1, 1))
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(len(shapes))).difference(test_inds))
    X_test = get_test_set(data_folder, target_size=params['img_size'])
    # Build a neural network model and train it with the training set
    print('Training CNN model...')
    scores = []
    for i in range(params['nb_trials']):
        print('Round #%i' % (i + 1))
        model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
        model.fit(X[train_inds], Y[train_inds], epochs=params['nb_epochs'],
                  shuffle=True, verbose=1, batch_size=params['batch_size'],
                  callbacks=[EarlyStopping(monitor='loss', patience=5)])
        score = evaluate_secondOrder(model, X_test, layer_num=-4,
                                     batch_size=params['batch_size'])
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
    tf.set_random_seed(0)
    main()