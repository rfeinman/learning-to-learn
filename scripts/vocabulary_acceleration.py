from __future__ import division, print_function

import os
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder

from learning2learn.util import evaluate_generalization
from learning2learn.util import train_test_split
from learning2learn.util import get_hidden_representations
from learning2learn.wrangle import get_train_test_parameters
from learning2learn.wrangle import build_test_trials_O2
from learning2learn.wrangle import build_train_set
from learning2learn.models import simple_cnn_multiout


def synthesize_data_multi(nb_shapes, nb_colors, nb_textures, nb_exemplars):
    shapes = []
    for i in range(nb_shapes):
        shapes.extend([i for _ in range(nb_exemplars)])
    colors = np.random.choice(range(nb_colors), size=nb_shapes*nb_exemplars)
    textures = np.random.choice(range(nb_textures), size=nb_shapes*nb_exemplars)
    df = pd.DataFrame(columns=['shape', 'color', 'texture'])
    df['shape'] = shapes
    df['color'] = colors
    df['texture'] = textures

    return df

def get_correct_classes(Y_train, Y_pred, nb_categories, threshold=0.8):
    # get predictions and ground truth
    y_true = Y_train.argmax(axis=1)
    y_pred = Y_pred.argmax(axis=1)
    # collect accuracies for each class
    classes = np.arange(nb_categories)
    accuracies = np.zeros_like(classes, dtype=np.float32)
    for i, c in enumerate(classes):
        ix = np.where(y_true == c)[0]
        nb_correct = len(np.where(y_pred[ix] == c)[0])
        accuracies[i] = nb_correct / len(ix)
    # collect vocabulary and return
    return np.where(accuracies >= threshold)[0]

def save_scores(epoch, o2_correct, vocabSize80_s, vocabSize80_t, logfile):
    df = pd.DataFrame()
    df['epoch'] = epoch
    df['o2_correct'] = o2_correct
    df['vocabSize80_s'] = vocabSize80_s
    df['vocabSize80_t'] = vocabSize80_t
    df.to_csv(logfile, index=False)

def evaluate_model(model, X_train, Y_train_s, Y_train_t, X_test, v80_s, v80_t):
    acc_o2 = evaluate_generalization(
        model, X_test, layer_num=-5,
        batch_size=128
    )
    Y_pred_s = get_hidden_representations(model, X_train, layer_num=-3,
                                          batch_size=128)
    Y_pred_t = get_hidden_representations(model, X_train, layer_num=-1,
                                          batch_size=128)
    correct80_s = get_correct_classes(
        Y_train_s, Y_pred_s, nb_categories=Y_train_s.shape[-1], threshold=0.8
    )
    correct80_t = get_correct_classes(
        Y_train_t, Y_pred_t, nb_categories=Y_train_t.shape[-1], threshold=0.8
    )
    v80_s = v80_s.union(correct80_s)
    v80_t = v80_t.union(correct80_t)

    return acc_o2, v80_s, v80_t

def get_data(nb_shapes, nb_colors, nb_textures, nb_exemplars):
    print('Building image datasets...')
    # Get the shape, color and texture features
    (shape_set_train, shape_set_test), \
    (color_set_train, color_set_test), \
    (texture_set_train, texture_set_test) = \
        get_train_test_parameters(images=True, img_size=200)
    if nb_shapes < 50:
        shape_set_train, _ = train_test_split(shape_set_train, 50-nb_shapes)
    if nb_colors < 50:
        color_set_train, _ = train_test_split(color_set_train, 50-nb_colors)
    if nb_textures < 50:
        texture_set_train, _ = train_test_split(
            texture_set_train, 50-nb_textures
        )
    # Get the training data
    df_train = synthesize_data_multi(
        nb_shapes, nb_colors, nb_textures, nb_exemplars
    )
    ohe = OneHotEncoder(sparse=False)
    Y_train_s = ohe.fit_transform(df_train['shape'].values.reshape(-1, 1))
    Y_train_c = ohe.fit_transform(df_train['color'].values.reshape(-1, 1))
    Y_train_t = ohe.fit_transform(df_train['texture'].values.reshape(-1, 1))
    # Build the training set of images
    X_train = build_train_set(
        df_train, shape_set_train,
        color_set_train, texture_set_train,
        shift_scale=20
    )
    # build the test set of images
    X_test = build_test_trials_O2(
        shape_set_test, color_set_test,
        texture_set_test, nb_trials=args.nb_test,
        shift_scale=20
    )

    return X_train, Y_train_s, Y_train_c, Y_train_t, X_test

def main():
    nb_shapes = int(np.round(args.shape_fraction*args.nb_categories))
    nb_colors = int(np.round(args.color_fraction*args.nb_categories))
    nb_textures = args.nb_categories - nb_shapes - nb_colors
    texture_fraction = 1. - args.shape_fraction - args.color_fraction
    print('nb_shapes: %i' % nb_shapes)
    print('nb_colors: %i' % nb_colors)
    print('nb_textures: %i' % nb_textures)
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        K.set_session(sess)
    # Set random seed
    np.random.seed(0)
    # create folder for results
    if os.path.isdir(args.save_path):
        warnings.warn('Removing old results folder of the same name!')
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)
    if not args.varying_data:
        print('Not varying the data across runs.')
        X_train, Y_train_s, Y_train_c, Y_train_t, X_test = \
            get_data(nb_shapes, nb_colors, nb_textures, args.nb_exemplars)
    for i in range(args.nb_runs):
        print('Run #%i' % i)
        if args.varying_data:
            print('Selecting data for run %i' % i)
            X_train, Y_train_s, Y_train_c, Y_train_t, X_test = \
                get_data(nb_shapes, nb_colors, nb_textures, args.nb_exemplars)
        v80_s = set([])  # keep track of shape vocab w/ threshold = 80
        v80_t = set([])  # keep track of texture vocab w/ threshold = 80
        save_file = os.path.join(
            args.save_path,
            'run_%0.2i.csv' % i
        )
        # initialize the model and compute initial metrics
        print('Initializing the model and computing initial metrics...')
        epoch = []
        o2_correct = []
        vocabSize80_s = []
        vocabSize80_t = []
        model = simple_cnn_multiout(
            input_shape=X_train.shape[1:],
            nb_shapes=Y_train_s.shape[-1],
            nb_colors=Y_train_c.shape[-1],
            nb_textures=Y_train_t.shape[-1],
            loss_weights=[args.shape_fraction, args.color_fraction,
                          texture_fraction]
        )
        acc_o2, v80_s, v80_t = evaluate_model(
            model, X_train, Y_train_s, Y_train_t,
            X_test, v80_s, v80_t
        )
        nb_correct = int(np.round(acc_o2*args.nb_test))
        print('initial # shape selections: %0.3f' % nb_correct)
        epoch.append(0)
        o2_correct.append(nb_correct)
        vocabSize80_s.append(len(v80_s))
        vocabSize80_t.append(len(v80_t))
        save_scores(epoch, o2_correct, vocabSize80_s, vocabSize80_t, save_file)
        for j in range(args.nb_epochs):
            print('Epoch #%i' % (j + 1))
            model.fit(
                [X_train],
                [Y_train_s, Y_train_c, Y_train_t],
                epochs=1,
                batch_size=args.batch_size,
                verbose=2
            )
            acc_o2, v80_s, v80_t = evaluate_model(
                model, X_train, Y_train_s, Y_train_t,
                X_test, v80_s, v80_t
            )
            nb_correct = int(np.round(acc_o2*args.nb_test))
            epoch.append(j+1)
            o2_correct.append(nb_correct)
            vocabSize80_s.append(len(v80_s))
            vocabSize80_t.append(len(v80_t))
            save_scores(
                epoch, o2_correct, vocabSize80_s, vocabSize80_t, save_file
            )
    if args.gpu_num is not None:
        K.clear_session()
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-sf', '--shape_fraction',
                        help='The fraction to separate by shape.',
                        required=False, type=float)
    parser.add_argument('-cf', '--color_fraction',
                        help='The fraction to separate by color.',
                        required=False, type=float)
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-sp', '--save_path',
                        help='The file path where results should be saved',
                        required=False, type=str)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use.',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use.',
                        required=False, type=int)
    parser.add_argument('-r', '--nb_runs',
                        help='The number of training runs.',
                        required=False, type=int)
    parser.add_argument('-t', '--nb_test',
                        help='The number of test trials each epoch.',
                        required=False, type=int)
    parser.add_argument('--varying_data',
                        help='Whether to vary the data accross each model run.',
                        required=False, action='store_true')
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(shape_fraction=0.6)
    parser.set_defaults(color_fraction=0.2)
    parser.set_defaults(nb_categories=15)
    parser.set_defaults(nb_exemplars=15)
    parser.set_defaults(save_path='../results/vocab_accel')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=10)
    parser.set_defaults(nb_runs=20)
    parser.set_defaults(nb_test=500)
    parser.set_defaults(varying_data=False)
    args = parser.parse_args()
    print(args)
    main()