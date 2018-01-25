from __future__ import division, print_function

import os
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import image
import matplotlib.path as mplpath
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split

from learning2learn.util import (build_vocab_training_set, evaluate_secondOrder,
                                 synthesize_data, shift_image)
from learning2learn.models import simple_cnn
from learning2learn.images import generate_random_shape
#from cnn_train_secondOrder import build_test_trials


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

def generate_colors():
    nb_colors = 64
    nb_bins = 4
    vals = np.linspace(0, 0.9, nb_bins)
    colors = np.zeros(shape=(nb_colors, 3))
    i = 0
    for r in vals:
        for g in vals:
            for b in vals:
                colors[i] = np.asarray([r, g, b])
                i += 1

    return colors

def adjust_contrast(img, factor):
    assert factor >= 1.
    img_p = 1. - img
    img_p /= factor
    img_p = 1. - img_p

    return img_p

def generate_image(shape, color, texture, target_size=(200, 200),
                   contrast_factor=1.):
    # Generate the base color
    img_color = np.ones(shape=target_size + (3,), dtype=np.float32) * color
    # Generate the base texture
    img_texture = image.load_img(
        '../data/textures/%s' % texture,
        target_size=target_size,
        interpolation='bicubic'
    )
    img_texture = image.img_to_array(img_texture) / 255.
    img_texture = img_texture[:, :, 0]
    img_texture = adjust_contrast(img_texture, contrast_factor)
    # Put it all together
    img = np.ones(shape=target_size + (4,), dtype=np.float32)
    img[:, :, :3] = img_color
    #img[:, :, 3] = img_texture
    # Cutout the shape
    p = mplpath.Path(shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not p.contains_point((i, j)):
                img[j, i, :] = np.ones_like(img[j, i])
    img = shift_image(img, img_size=target_size, scale=20)
    img[:, :, 3] = img_texture

    return img

def generate_image_wrapper(tup):
    return generate_image(tup[0], tup[1], tup[2])

def make_trial(shape_set, color_set, texture_set, target_size=(200, 200),
               contrast_factor=1.):
    # randomly select 3 of each feature
    s1, s2, s3 = np.random.choice(range(len(shape_set)), 3, replace=False)
    c1, c2, c3 = np.random.choice(range(len(color_set)), 3, replace=False)
    shape1, shape2, shape3 = shape_set[s1], shape_set[s2], shape_set[s3]
    color1, color2, color3 = color_set[c1], color_set[c2], color_set[c3]
    texture1, texture2, texture3 = np.random.choice(texture_set, 3,
                                                    replace=False)
    # generate the trial images
    baseline = generate_image(shape1, color1, texture1, target_size,
                              contrast_factor)
    shape_match = generate_image(shape1, color2, texture2, target_size,
                                 contrast_factor)
    color_match = generate_image(shape2, color1, texture3, target_size,
                                 contrast_factor)
    texture_match = generate_image(shape3, color3, texture1, target_size,
                                   contrast_factor)

    return np.asarray([baseline, shape_match, color_match, texture_match])

def make_trial_wrapper(tup):
    return make_trial(tup[0], tup[1], tup[2], tup[3], tup[4])

def build_test_trials(shape_set, color_set, texture_set, nb_trials,
                      target_size=(200, 200), contrast_factor=1.):
    tups = [(shape_set, color_set, texture_set, target_size, contrast_factor)
            for _ in range(nb_trials)]
    p = mp.Pool()
    trials = p.map(make_trial_wrapper, tups)

    return np.concatenate(trials)

def compute_area(shape, img_size):
    area = 0
    p = mplpath.Path(shape)
    for i in range(img_size):
        for j in range(img_size):
            if p.contains_point((i, j)):
                area += 1

    return area

def train_test_split(x, test_size):
    step = int(np.ceil(len(x) / test_size)) - 1
    ix = list(range(len(x)))
    ix_test = [i * step for i in range(test_size)]
    ix_train = list(set(ix).difference(ix_test))
    x_train = [x[i] for i in ix_train]
    x_test = [x[i] for i in ix_test]

    return x_train, x_test

def get_secondOrder_data(df_train, nb_test_trials=1000):
    # count the number of categories in the training set
    nb_train = len(np.unique(df_train['shape']))
    # we have 58 textures, so that is our limiting factor
    assert nb_train < 58
    nb_test = 58 - nb_train
    # get the 58 shapes
    shape_set = [generate_random_shape(0, 200, 0, 200, 40) for _ in range(58)]
    shape_set = sorted(shape_set, key=lambda x: compute_area(x, 200))
    # get the 58 colors
    color_set = generate_colors()
    ix = np.sort(np.random.choice(range(len(color_set)), 58, replace=False))
    color_set = color_set[ix]
    # get the 58 textures
    texture_set = sorted(
        [file for file in os.listdir('../data/textures') if
         file.endswith('tiff')]
    )
    # perform the train/test splits
    shape_set_train, shape_set_test = train_test_split(shape_set,
                                                       test_size=nb_test)
    color_set_train, color_set_test = train_test_split(color_set,
                                                       test_size=nb_test)
    texture_set_train, texture_set_test = train_test_split(texture_set,
                                                           test_size=nb_test)
    # build the training set of images
    print('Building training set...')
    tups = []
    for i in range(len(df_train)):
        s, c, t = df_train.iloc[i]
        tups.append(
            (shape_set_train[s], color_set_train[c], texture_set_train[t])
        )
    p = mp.Pool()
    X_train = p.map(generate_image_wrapper, tups)
    p.close()
    p.join()
    X_train = np.asarray(X_train)
    # build the test set trials
    print('Building test trials...')
    X_test = build_test_trials(shape_set_test, color_set_test, texture_set_test,
                               nb_test_trials)

    return X_train, X_test

def evaluate_model(model, X_train, Y_train, X_test):
    loss, acc = model.evaluate(X_train, Y_train, batch_size=256, verbose=0)
    acc2 = evaluate_secondOrder(
        model, X_test, layer_num=-4,
        batch_size=256
    )
    vs50, vs80 = compute_vocab_size(model, X_train, Y_train)

    return loss, acc, acc2, vs50, vs80

def main():
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        K.set_session(sess)
    # load the dataset
    # print('Loading the data...')
    # data_folder = '../data/images_generated2/'
    # X, labels = build_vocab_training_set(
    #     data_folder=data_folder,
    #     nb_exemplars=15,
    #     nb_categories=50,
    #     shape_fraction=args.shape_fraction,
    #     color_fraction=args.color_fraction,
    #     shift=False
    # )
    # test_folder = os.path.join(data_folder, 'test/')
    # X_test = build_test_trials(test_folder, nb_trials=1000,
    #                            target_size=(200, 200), shift=False)
    df_train, labels = synthesize_data(nb_categories=50, nb_exemplars=15)
    X_train, X_test = get_secondOrder_data(df_train, nb_test_trials=1000)
    # dummy code the labels
    ohe = OneHotEncoder(sparse=False)
    Y_train = ohe.fit_transform(labels.reshape(-1, 1))
    # initialize the model and compute initial metrics
    print('Initializing the model and computing initial metrics...')
    epoch = []
    trainLoss = []
    trainAcc = []
    secondOrderAcc = []
    vocabSize50 = []
    vocabSize80 = []
    model = simple_cnn(input_shape=X_train.shape[1:],
                       nb_classes=Y_train.shape[-1])
    loss, acc, acc2, vs50, vs80 = evaluate_model(model, X_train, Y_train,
                                                 X_test)
    print('initial shape bias: %0.3f' % acc2)
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
            X_train, Y_train, epochs=1,
            shuffle=True,
            verbose=1, batch_size=32
        )
        loss, acc, acc2, vs50, vs80 = evaluate_model(model, X_train, Y_train,
                                                     X_test)
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
    parser.set_defaults(save_path='../results/vocab_log.csv')
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    # Set random seeds
    np.random.seed(0)
    tf.set_random_seed(0)
    main()