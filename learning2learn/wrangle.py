from __future__ import division
import os
import itertools
import math
import random
import multiprocessing as mp
import pandas as pd
import numpy as np

from learning2learn.util import train_test_split
from learning2learn.images import (generate_image, generate_image_wrapper,
                                   generate_colors, generate_random_shape,
                                   compute_area)


def synthesize_data(nb_categories, nb_exemplars):
    labels = []
    shapes = []
    for i in range(nb_categories):
        labels.extend([i for _ in range(nb_exemplars)])
        shapes.extend([i for _ in range(nb_exemplars)])
    textures = np.random.permutation(shapes)
    colors = np.random.permutation(shapes)
    df = pd.DataFrame(columns=['shape', 'color', 'texture'])
    df['shape'] = shapes
    df['color'] = colors
    df['texture'] = textures

    return df, pd.Series(labels)

def generate_bit_dictionary(nb_categories, nb_bits=20):
    min_bits = int(np.ceil(math.log(nb_categories, 2)))
    assert nb_bits >= min_bits
    # the candidate vectors
    candidates = np.asarray(
        [seq for seq in itertools.product([0, 1], repeat=nb_bits)]
    )
    # the candidates that we will use are randomly selected
    ix = np.random.choice(
        range(len(candidates)), nb_categories, replace=False
    )

    return candidates[ix]

def build_train_set_bits(df_train, shape_set, color_set, texture_set):
    shape = np.asarray([shape_set[elt] for elt in df_train['shape']])
    color = np.asarray([color_set[elt] for elt in df_train['color']])
    texture = np.asarray([texture_set[elt] for elt in df_train['texture']])

    return np.concatenate((shape, color, texture), axis=1)

def build_test_trials_O1_bits(
        df_train, shape_set_train, shape_set_test, color_set_train,
        color_set_test, texture_set_train, texture_set_test, nb_trials
):
    nb_bits = len(shape_set_train[0])
    X_test = np.zeros((4 * nb_trials, 3 * nb_bits))
    for i in range(nb_trials):
        # First sample the baseline from the training set
        ix = np.random.choice(range(len(df_train)), 1)[0]
        s1, c1, t1 = df_train.iloc[ix]
        shape1 = shape_set_train[s1]
        color1 = color_set_train[c1]
        texture1 = texture_set_train[t1]
        # Now sample the 2 novel colors and textures that will be used
        s2, s3 = np.random.choice(range(len(shape_set_test)), 2, replace=False)
        shape2, shape3 = shape_set_test[s2], shape_set_test[s3]
        c2, c3 = np.random.choice(range(len(color_set_test)), 2, replace=False)
        color2, color3 = color_set_test[c2], color_set_test[c3]
        t2, t3 = np.random.choice(range(len(texture_set_test)), 2,
                                  replace=False)
        texture2, texture3 = texture_set_test[t2], texture_set_test[t3]
        # Now build the samples
        baseline = np.concatenate((shape1, color1, texture1))
        shape_match = np.concatenate((shape1, color2, texture2))
        color_match = np.concatenate((shape2, color1, texture3))
        texture_match = np.concatenate((shape3, color3, texture1))
        # Collect trial and return
        X_test[4 * i:4 * (i + 1)] = np.asarray(
            [baseline, shape_match, color_match, texture_match])

    return X_test

def build_test_trials_O2_bits(shape_set, color_set, texture_set, nb_trials):
    nb_bits = len(shape_set[0])
    X_test = np.zeros((4 * nb_trials, 3 * nb_bits))
    for i in range(nb_trials):
        # randomly select 3 of each feature
        s1, s2, s3 = np.random.choice(range(len(shape_set)), 3, replace=False)
        c1, c2, c3 = np.random.choice(range(len(color_set)), 3, replace=False)
        t1, t2, t3 = np.random.choice(range(len(texture_set)), 3, replace=False)
        shape1, shape2, shape3 = shape_set[s1], shape_set[s2], shape_set[s3]
        color1, color2, color3 = color_set[c1], color_set[c2], color_set[c3]
        texture1, texture2, texture3 = texture_set[c1], texture_set[c2], \
                                       texture_set[c3]
        # Now build the samples
        baseline = np.concatenate((shape1, color1, texture1))
        shape_match = np.concatenate((shape1, color2, texture2))
        color_match = np.concatenate((shape2, color1, texture3))
        texture_match = np.concatenate((shape3, color3, texture1))
        # Collect trial and return
        X_test[4 * i:4 * (i + 1)] = np.asarray(
            [baseline, shape_match, color_match, texture_match])

    return X_test

def build_train_set(
        df_train, shape_set, color_set, texture_set, target_size=(200, 200),
        shift_scale=20
):
    tups = []
    for i in range(len(df_train)):
        s, c, t = df_train.iloc[i]
        tups.append(
            (shape_set[s], color_set[c], texture_set[t], target_size,
             shift_scale)
        )
    p = mp.Pool()
    X_train = p.map(generate_image_wrapper, tups)
    p.close()
    p.join()

    return np.asarray(X_train)

def make_trial_O1(
        df_train, shape_set_train, shape_set_test, color_set_train,
        color_set_test, texture_set_train, texture_set_test,
        target_size=(200, 200), shift_scale=20
):
    # First sample the baseline from the training set
    i = np.random.choice(range(len(df_train)), 1)[0]
    s1, c1, t1 = df_train.iloc[i]
    shape1 = shape_set_train[s1]
    color1 = color_set_train[c1]
    texture1 = texture_set_train[t1]
    # Now sample the 2 novel colors and textures that will be used
    s2, s3 = np.random.choice(range(len(shape_set_test)), 2, replace=False)
    shape2, shape3 = shape_set_test[s2], shape_set_test[s3]
    c2, c3 = np.random.choice(range(len(color_set_test)), 2, replace=False)
    color2, color3 = color_set_test[c2], color_set_test[c3]
    t2, t3 = np.random.choice(range(len(texture_set_test)), 2, replace=False)
    texture2, texture3 = texture_set_test[t2], texture_set_test[t3]
    # Now build the images
    baseline = generate_image(
        shape1, color1, texture1, target_size, shift_scale
    )
    shape_match = generate_image(
        shape1, color2, texture2, target_size, shift_scale
    )
    color_match = generate_image(
        shape2, color1, texture3, target_size, shift_scale
    )
    texture_match = generate_image(
        shape3, color3, texture1, target_size, shift_scale
    )
    # Collect trial and return
    return np.asarray([baseline, shape_match, color_match, texture_match])

def make_trial_O1_wrapper(tup):
    seed = random.randint(0, 1e7)
    np.random.seed(seed)
    trial = make_trial_O1(
        tup[0], tup[1], tup[2], tup[3], tup[4], tup[5],
        tup[6], tup[7], tup[8]
    )

    return trial

def build_test_trials_O1(
        df_train, shape_set_train, shape_set_test, color_set_train,
        color_set_test, texture_set_train, texture_set_test, nb_trials,
        target_size=(200, 200), shift_scale=20
):
    tups = [(df_train, shape_set_train, shape_set_test, color_set_train,
             color_set_test, texture_set_train, texture_set_test,
             target_size, shift_scale)
            for _ in range(nb_trials)]
    p = mp.Pool()
    trials = p.map(make_trial_O1_wrapper, tups)
    p.close()
    p.join()

    return np.concatenate(trials)

def make_trial_O2(
        shape_set, color_set, texture_set, target_size=(200, 200),
        shift_scale=20
):
    # randomly select 3 of each feature
    s1, s2, s3 = np.random.choice(range(len(shape_set)), 3, replace=False)
    c1, c2, c3 = np.random.choice(range(len(color_set)), 3, replace=False)
    shape1, shape2, shape3 = shape_set[s1], shape_set[s2], shape_set[s3]
    color1, color2, color3 = color_set[c1], color_set[c2], color_set[c3]
    texture1, texture2, texture3 = np.random.choice(texture_set, 3,
                                                    replace=False)
    # generate the trial images
    baseline = generate_image(shape1, color1, texture1, target_size,
                              shift_scale)
    shape_match = generate_image(shape1, color2, texture2, target_size,
                                 shift_scale)
    color_match = generate_image(shape2, color1, texture3, target_size,
                                 shift_scale)
    texture_match = generate_image(shape3, color3, texture1, target_size,
                                   shift_scale)

    return np.asarray([baseline, shape_match, color_match, texture_match])

def make_trial_O2_wrapper(tup):
    # since trials are randomly selected, we want a different random seed
    # for each process
    seed = random.randint(0, 1e7)
    np.random.seed(seed)
    return make_trial_O2(tup[0], tup[1], tup[2], tup[3], tup[4])

def build_test_trials_O2(
        shape_set, color_set, texture_set, nb_trials, target_size=(200, 200),
        shift_scale=20
):
    tups = [(shape_set, color_set, texture_set, target_size, shift_scale)
            for _ in range(nb_trials)]
    p = mp.Pool()
    trials = p.map(make_trial_O2_wrapper, tups)
    p.close()
    p.join()

    return np.concatenate(trials)

def get_train_test_parameters(images=True, img_size=200, nb_bits=20):
    # we have 58 textures, so that is our limiting factor. We will use 50 for
    # training and hold out 8 for test.
    nb_train = 50
    nb_test = 8
    if images:
        # get the set of shapes
        shape_set = [generate_random_shape(0, img_size, 0, img_size,
                                           int(img_size/5.))
                     for _ in range(nb_train+nb_test)]
        shape_set = sorted(shape_set, key=lambda x: compute_area(x, img_size))
        # get the set of colors
        color_set = generate_colors()
        ix = np.sort(np.random.choice(range(len(color_set)), nb_train+nb_test,
                                  replace=False))
        color_set = color_set[ix]
        # get the set of textures
        texture_set = sorted(
            [file for file in os.listdir('../data/textures') if
             file.endswith('tiff')]
        )
    else:
        # get the set of shapes, colors and textures
        shape_set = generate_bit_dictionary(nb_train+nb_test, nb_bits)
        color_set = generate_bit_dictionary(nb_train+nb_test, nb_bits)
        texture_set = generate_bit_dictionary(nb_train+nb_test, nb_bits)
    # perform the train/test splits
    shape_set_train, shape_set_test = train_test_split(shape_set,
                                                       test_size=nb_test)
    color_set_train, color_set_test = train_test_split(color_set,
                                                       test_size=nb_test)
    texture_set_train, texture_set_test = train_test_split(texture_set,
                                                           test_size=nb_test)

    return (shape_set_train, shape_set_test), \
           (color_set_train, color_set_test), \
           (texture_set_train, texture_set_test)