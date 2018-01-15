from __future__ import division, print_function

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder

from learning2learn.util import build_vocab_training_set, evaluate_secondOrder
from learning2learn.models import simple_cnn
from cnn_train_secondOrder import build_test_trials


def compute_vocab_size(model, X, Y, batch_size=64):
    # get predictions and ground truth
    y_true = Y.argmax(axis=1)
    y_pred = model.predict_classes(X, batch_size=batch_size)
    # collect accuracies for each class
    classes = np.unique(y_true)
    accuracies = np.zeros_like(classes, dtype=np.float32)
    for i, c in enumerate(classes):
        ix = np.where(y_true == c)[0]
        nb_correct = len(np.where(y_pred[ix] == c)[0])
        accuracies[i] = nb_correct / len(ix)
    # Compute vocab size.
    vocab_size50 = len(np.where(accuracies >= 0.5)[0])
    vocab_size80 = len(np.where(accuracies >= 0.8)[0])

    return vocab_size50, vocab_size80

def save_scores(epoch, trainLoss, trainAcc, secondOrderAcc, vocabSize50,
                vocabSize80, logfile):
    df = pd.DataFrame()
    df['epoch'] = epoch
    df['trainLoss'] = trainLoss
    df['trainAcc'] = trainAcc
    df['secondOrderAcc'] = secondOrderAcc
    df['vocabSize50'] = vocabSize50
    df['vocabSize80'] = vocabSize80
    df.to_csv(logfile, index=False)

def main():
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        K.set_session(sess)
    # load the dataset
    print('Loading the data...')
    data_folder = '../data/images_generated/'
    X, labels = build_vocab_training_set(
        data_folder=data_folder,
        nb_exemplars=15,
        nb_categories=50,
        shape_fraction=args.shape_fraction,
        color_fraction=args.color_fraction,
        shift=True
    )
    test_folder = os.path.join(data_folder, 'test/')
    X_test = build_test_trials(test_folder, nb_trials=1000,
                               target_size=(200, 200), shift=True)
    # dummy code the labels
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(labels.reshape(-1, 1))
    # initialize the model and compute initial metrics
    print('Initializing the model and computing initial metrics...')
    epoch = []
    trainLoss = []
    trainAcc = []
    secondOrderAcc = []
    vocabSize50 = []
    vocabSize80 = []
    model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
    loss, acc = model.evaluate(X, Y, batch_size=64, verbose=0)
    acc2 = evaluate_secondOrder(
        model, X_test, layer_num=-4,
        batch_size=64
    )
    vs50, vs80 = compute_vocab_size(model, X, Y)
    epoch.append(0)
    trainLoss.append(loss)
    trainAcc.append(acc)
    secondOrderAcc.append(acc2)
    vocabSize50.append(vs50)
    vocabSize80.append(vs80)
    save_scores(epoch, trainLoss, trainAcc, secondOrderAcc, vocabSize50,
                vocabSize80, args.save_path)
    print('Training the model...')
    for i in range(args.nb_epochs):
        print('Epoch #%i' % (i + 1))
        model.fit(
            X, Y, epochs=1,
            shuffle=True,
            verbose=1, batch_size=32
        )
        loss, acc = model.evaluate(X, Y, batch_size=64, verbose=0)
        acc2 = evaluate_secondOrder(
            model, X_test, layer_num=-4,
            batch_size=64
        )
        vs50, vs80 = compute_vocab_size(model, X, Y)
        epoch.append(i + 1)
        trainLoss.append(loss)
        trainAcc.append(acc)
        secondOrderAcc.append(acc2)
        vocabSize50.append(vs50)
        vocabSize80.append(vs80)
        save_scores(epoch, trainLoss, trainAcc, secondOrderAcc, vocabSize50,
                    vocabSize80, args.save_path)


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
    # parser.add_argument('-th', '--threshold',
    #                     help='The accuracy threshold for word acquisition.',
    #                     required=False, type=float)
    parser.add_argument('-sp', '--save_path',
                        help='The file path where results should be saved',
                        required=False, type=str)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use.',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use.',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(shape_fraction=0.6)
    parser.set_defaults(color_fraction=0.2)
    # parser.set_defaults(threshold=0.8)
    parser.set_defaults(save_path='../results/vocab_log.csv')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    # Set random seeds
    np.random.seed(0)
    main()