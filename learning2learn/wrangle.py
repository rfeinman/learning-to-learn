from __future__ import division
import os
import itertools
import warnings
import math
import multiprocessing as mp
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from learning2learn.util import train_test_split
from learning2learn.images import (generate_image, generate_image_wrapper,
                                   generate_colors, generate_random_shape,
                                   compute_area)


def generate_dictionary(column, nb_bits=None):
    """
    Observe the set of all values in a given feature column and create a
    randomly-assigned dictionary with one bitmap vector for each value.
    TODO: update this so that bitmap vectors are orthogonal
    :param column: (Pandas Series) a feature column from your data set
    :param nb_bits: (int) the number of bits to use in the bitmap vectors. If
    None (default), the number is chosen based on the set of unique values in
    the feature column.
    :return: (dict) a dictionary containing one bitmap vector per feature value.
    """
    val_set = np.unique(column)
    nb_vals = val_set.shape[0]
    if nb_bits is None:
        # compute the required number of bits based on the
        # number of values. Let's add 2 bits for extra space.
        nb_bits = int(round(math.log(nb_vals, 2))) + 2
    else:
        assert type(nb_bits) == int
    # the candidate vectors
    candidates = np.asarray([seq for seq in itertools.product([0,1],
                                                              repeat=nb_bits)])
    # the candidates that we will use are randomly selected
    inds = np.random.choice(range(len(candidates)), nb_vals, replace=False)
    # now select the vectors
    vectors = candidates[inds]

    return {val_set[i]: vectors[i] for i in range(nb_vals)}

def convert_column(column, nb_bits=None):
    dictionary = generate_dictionary(column, nb_bits)

    return np.asarray([dictionary[elt] for elt in column])

def preprocess_data(df, labels, one_hot=False, nb_bits=None):
    # check on form of the dataframe
    cols = df.columns.tolist()
    assert len(cols) == 3
    assert 'shape' in cols and 'color' in cols and 'texture' in cols
    # warn if nb_bits is provided but will not be used
    if one_hot and nb_bits is not None:
        warnings.warn('nb_bits parameter is not used when one_hot=True.')
    # encode the categorical variables
    if one_hot:
        # one-hot-encode the categorical variables in the data set
        df = pd.get_dummies(df, columns=['shape', 'color', 'texture'])
        X = df.values
    else:
        # select bitmap vectors for each categorical variable
        shape = convert_column(df['shape'], nb_bits)
        color = convert_column(df['color'], nb_bits)
        texture = convert_column(df['texture'], nb_bits)
        X = np.concatenate((shape, color, texture), axis=1)
    # turn label words into number indices
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # one-hot-encode the labels so that they work with softmax output
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(y.reshape(-1, 1))

    return X, Y

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

def synthesize_new_data(nb_categories):
    """
    Synthesize new object data for the second order generalization test.
    Groupings of 4 samples are generated: first, a baseline example of the new
    object category, and then 3 comparison samples. One of the comparison
    samples maintains the same shape as the baseline, another the same
    color, and another the same texture. For each, the other features are
    different from the baseline.
    :param nb_categories: (int) The number of categories in our original data
                            set
    :return: (Pandas DataFrame, Pandas Series) The data features and labels
    """
    # Create the first grouping
    a = np.asarray([[nb_categories, nb_categories, nb_categories],
                    [nb_categories, nb_categories+1, nb_categories+1],
                    [nb_categories+1, nb_categories, nb_categories+2],
                    [nb_categories+2, nb_categories+2, nb_categories]])
    # Loop through, incrementing grouping by 3 each time and stacking them all
    # on top of each other
    dfs = []
    labels = []
    for i in range(nb_categories):
        dfs.append(pd.DataFrame(a+3*i, columns=['shape', 'color', 'texture']))
        labels.extend([nb_categories+3*i for _ in range(4)])

    return pd.concat(dfs), pd.Series(labels)

def build_train_set(df_train, shape_set, color_set, texture_set,
                    target_size=(200, 200), contrast_factor=1.):
    tups = []
    for i in range(len(df_train)):
        s, c, t = df_train.iloc[i]
        tups.append(
            (shape_set[s], color_set[c], texture_set[t], target_size,
             contrast_factor)
        )
    p = mp.Pool()
    X_train = p.map(generate_image_wrapper, tups)
    p.close()
    p.join()

    return np.asarray(X_train)

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

def get_train_test_parameters(img_size=200):
    # we have 58 textures, so that is our limiting factor. We will use 50 for
    # training and hold out 8 for test.
    nb_train = 50
    nb_test = 8
    # get the 58 shapes
    shape_set = [generate_random_shape(0, img_size, 0, img_size,
                                       int(img_size/5.))
                 for _ in range(nb_train+nb_test)]
    shape_set = sorted(shape_set, key=lambda x: compute_area(x, img_size))
    # get the 58 colors
    color_set = generate_colors()
    ix = np.sort(np.random.choice(range(len(color_set)), nb_train+nb_test,
                                  replace=False))
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

    return (shape_set_train, shape_set_test), \
           (color_set_train, color_set_test), \
           (texture_set_train, texture_set_test)