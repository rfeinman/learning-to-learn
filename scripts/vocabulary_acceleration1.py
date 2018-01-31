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

from learning2learn.util import evaluate_generalization, train_test_split
from learning2learn.wrangle import (synthesize_data, get_train_test_parameters,
                                    build_test_trials_order2, build_train_set)
from learning2learn.models import simple_cnn


def get_correct_classes(model, X_train, Y_train, nb_categories, threshold=0.8):
    # get predictions and ground truth
    y_true = Y_train.argmax(axis=1)
    y_pred = model.predict_classes(X_train, batch_size=128)
    # collect accuracies for each class
    classes = np.arange(nb_categories)
    accuracies = np.zeros_like(classes, dtype=np.float32)
    for i, c in enumerate(classes):
        ix = np.where(y_true == c)[0]
        nb_correct = len(np.where(y_pred[ix] == c)[0])
        accuracies[i] = nb_correct / len(ix)
    # collect vocabulary and return
    return np.where(accuracies >= threshold)[0]

def save_scores(epoch, trainLoss, trainAcc, o2_correct, vocabSize50,
                vocabSize80, logfile):
    df = pd.DataFrame()
    df['epoch'] = epoch
    df['trainLoss'] = trainLoss
    df['trainAcc'] = trainAcc
    df['o2_correct'] = o2_correct
    df['vocabSize50'] = vocabSize50
    df['vocabSize80'] = vocabSize80
    df.to_csv(logfile, index=False)

def evaluate_model(model, X_train, Y_train, X_test, v50, v80):
    loss, acc = model.evaluate(X_train, Y_train, batch_size=256, verbose=0)
    acc2 = evaluate_generalization(
        model, X_test, layer_num=-4,
        batch_size=256
    )
    correct50 = get_correct_classes(
        model, X_train, Y_train, nb_categories=Y_train.shape[-1], threshold=0.5
    )
    correct80 = get_correct_classes(
        model, X_train, Y_train, nb_categories=Y_train.shape[-1], threshold=0.8
    )
    v50 = v50.union(correct50)
    v80 = v80.union(correct80)

    return loss, acc, acc2, v50, v80

def get_data(nb_categories, nb_exemplars):
    print('Building image datasets...')
    # Get the training data
    df_train, labels = synthesize_data(
        nb_categories=nb_categories,
        nb_exemplars=nb_exemplars
    )
    ohe = OneHotEncoder(sparse=False)
    Y_train = ohe.fit_transform(labels.values.reshape(-1, 1))
    # Get the shape, color and texture features
    (shape_set_train, shape_set_test), \
    (color_set_train, color_set_test), \
    (texture_set_train, texture_set_test) = \
        get_train_test_parameters(images=True, img_size=200)
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
    # Build the training set of images
    X_train = build_train_set(
        df_train, shape_set_train,
        color_set_train, texture_set_train,
        shift_scale=20
    )
    # build the test set of images
    X_test = build_test_trials_order2(
        shape_set_test, color_set_test,
        texture_set_test, nb_trials=args.nb_test,
        shift_scale=20
    )

    return X_train, Y_train, X_test

def main():
    assert args.nb_categories <= 50
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
        X_train, Y_train, X_test = get_data(args.nb_categories,
                                            args.nb_exemplars)
    for i in range(args.nb_runs):
        if args.varying_data:
            print('Selecting data for run %i' % i)
            X_train, Y_train, X_test = get_data(args.nb_categories,
                                                args.nb_exemplars)
        # enforce a maximum batch size to avoid doing full-batch GD
        batch_size = min(
            args.batch_size,
            int(np.floor(args.nb_categories*args.nb_exemplars / 5))
        )
        v50 = set([])  # keep track of vocab w/ threshold = 50
        v80 = set([])  # keep track of vocab w/ threshold = 80
        save_file = os.path.join(
            args.save_path,
            'run_%0.2i.csv' % i
        )
        # initialize the model and compute initial metrics
        print('Initializing the model and computing initial metrics...')
        epoch = []
        trainLoss = []
        trainAcc = []
        o2_correct = []
        vocabSize50 = []
        vocabSize80 = []
        model = simple_cnn(
            input_shape=X_train.shape[1:],
            nb_classes=Y_train.shape[-1]
        )
        loss, acc, acc2, v50, v80 = evaluate_model(
            model, X_train, Y_train, X_test, v50, v80
        )
        nb_correct = int(np.round(acc2*args.nb_test))
        print('initial shape bias: %0.3f' % acc2)
        epoch.append(0)
        trainLoss.append(loss)
        trainAcc.append(acc)
        o2_correct.append(nb_correct)
        vocabSize50.append(len(v50))
        vocabSize80.append(len(v80))
        save_scores(
            epoch, trainLoss, trainAcc, o2_correct,
            vocabSize50, vocabSize80, save_file
        )
        for j in range(args.nb_epochs):
            print('Epoch #%i' % (j + 1))
            model.fit(
                X_train, Y_train, epochs=1,
                shuffle=True,
                verbose=2, batch_size=batch_size
            )
            loss, acc, acc2, v50, v80 = evaluate_model(
                model, X_train, Y_train, X_test, v50, v80
            )
            nb_correct = int(np.round(acc2*args.nb_test))
            epoch.append(j + 1)
            trainLoss.append(loss)
            trainAcc.append(acc)
            o2_correct.append(nb_correct)
            vocabSize50.append(len(v50))
            vocabSize80.append(len(v80))
            save_scores(
                epoch, trainLoss, trainAcc, o2_correct,
                vocabSize50, vocabSize80, save_file
            )
    if args.gpu_num is not None:
        K.clear_session()
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    # parser.add_argument('-sf', '--shape_fraction',
    #                     help='The fraction to separate by shape.',
    #                     required=False, type=float)
    # parser.add_argument('-cf', '--color_fraction',
    #                     help='The fraction to separate by color.',
    #                     required=False, type=float)
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
    # parser.set_defaults(shape_fraction=1.0)
    # parser.set_defaults(color_fraction=0.0)
    parser.set_defaults(nb_categories=15)
    parser.set_defaults(nb_exemplars=15)
    parser.set_defaults(save_path='../results/vocab_log')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(nb_runs=5)
    parser.set_defaults(nb_test=100)
    parser.set_defaults(varying_data=False)
    args = parser.parse_args()
    print('nb_categories: %i, nb_exemplars: %i, nb_runs: %i, nb_test: %i, '
          'varying_data: %r' %
          (args.nb_categories, args.nb_exemplars, args.nb_runs, args.nb_test,
           args.varying_data))
    main()