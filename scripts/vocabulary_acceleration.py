from __future__ import division, print_function

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import SGD

from learning2learn.util import evaluate_generalization
from learning2learn.wrangle import (synthesize_data, get_train_test_parameters,
                                    build_test_trials_order2, build_train_set)
from learning2learn.models import simple_cnn

# def build_model(layers):
#     """
#     Takes in a list of Keras layers and returns an initialized
#     Keras Sequential model with those layers.
#     :param layers: (list) the list of Keras layers for the model
#     :return: (keras Sequential) a compiled Keras model
#     """
#     model = Sequential()
#     for layer in layers:
#         model.add(layer)
#     model.compile(
#         loss='categorical_crossentropy',
#         optimizer=SGD(lr=0.01, momentum=0.8),
#         metrics=['accuracy']
#     )
#
#     return model
#
# def simple_cnn(input_shape, nb_classes):
#     layers = [
#         # Conv, Pool
#         Conv2D(32, (5, 5), padding='same', input_shape=input_shape),
#         Activation('relu'),
#         MaxPooling2D(pool_size=(5, 5)),
#         # Conv, Pool
#         Conv2D(64, (5, 5), padding='same'),
#         Activation('relu'),
#         MaxPooling2D(pool_size=(5, 5)),
#         # Flatten
#         Flatten(),
#         # Hidden layer
#         Dropout(0.5),
#         Dense(2048),
#         Activation('relu'),
#         # Output layer
#         Dropout(0.5),
#         Dense(nb_classes),
#         Activation('softmax')
#     ]
#
#     return build_model(layers)

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

def evaluate_model(model, X_train, Y_train, X_test):
    loss, acc = model.evaluate(X_train, Y_train, batch_size=256, verbose=0)
    acc2 = evaluate_generalization(
        model, X_test, layer_num=-4,
        batch_size=256
    )
    vs50, vs80 = compute_vocab_size(model, X_train, Y_train)

    return loss, acc, acc2, vs50, vs80

def main():
    # Set random seeds
    np.random.seed(0)
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        K.set_session(sess)
    # get parameters
    (shape_set_train, shape_set_test), \
    (color_set_train, color_set_test), \
    (texture_set_train, texture_set_test) = \
        get_train_test_parameters()
    # build train set
    df_train, labels = synthesize_data(nb_categories=50, nb_exemplars=15)
    print('Building the training set...')
    X_train = build_train_set(
        df_train, shape_set_train,
        color_set_train, texture_set_train
    )
    ohe = OneHotEncoder(sparse=False)
    Y_train = ohe.fit_transform(labels.reshape(-1, 1))
    # build test set
    print('Building test trials...')
    X_test = build_test_trials_order2(
        shape_set_test, color_set_test,
        texture_set_test, nb_trials=2000
    )
    # initialize the model and compute initial metrics
    print('Initializing the model and computing initial metrics...')
    epoch = []
    trainLoss = []
    trainAcc = []
    secondOrderAcc = []
    vocabSize50 = []
    vocabSize80 = []
    model = simple_cnn(
        input_shape=X_train.shape[1:],
        nb_classes=Y_train.shape[-1]
    )
    loss, acc, acc2, vs50, vs80 = evaluate_model(
        model, X_train, Y_train, X_test
    )
    print('initial shape bias: %0.3f' % acc2)
    epoch.append(0)
    trainLoss.append(loss)
    trainAcc.append(acc)
    secondOrderAcc.append(acc2)
    vocabSize50.append(vs50)
    vocabSize80.append(vs80)
    save_scores(
        epoch, trainLoss, trainAcc, secondOrderAcc,
        vocabSize50, vocabSize80, args.save_path
    )
    print('Training the model...')
    for i in range(args.nb_epochs):
        print('Epoch #%i' % (i + 1))
        model.fit(
            X_train, Y_train, epochs=1,
            shuffle=True,
            verbose=1, batch_size=32
        )
        loss, acc, acc2, vs50, vs80 = evaluate_model(
            model, X_train, Y_train, X_test
        )
        epoch.append(i + 1)
        trainLoss.append(loss)
        trainAcc.append(acc)
        secondOrderAcc.append(acc2)
        vocabSize50.append(vs50)
        vocabSize80.append(vs80)
        save_scores(
            epoch, trainLoss, trainAcc, secondOrderAcc,
            vocabSize50, vocabSize80, args.save_path
        )

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
    parser.set_defaults(shape_fraction=1.0)
    parser.set_defaults(color_fraction=0.0)
    parser.set_defaults(save_path='../results/vocab_log.csv')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()