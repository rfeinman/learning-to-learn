from __future__ import division
import itertools
import warnings
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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

def get_train_test_inds(nb_categories, nb_exemplars, nb_samples, nb_test=1):
    """

    :param nb_categories:
    :param nb_exemplars:
    :param nb_shapes:
    :param nb_test:
    :return:
    """
    test_inds = []
    for i in range(nb_categories):
        bottom = i * (nb_exemplars + nb_test)
        top = bottom + nb_test
        test_inds.extend(range(bottom, top))
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(nb_samples)).difference(test_inds))

    return train_inds, test_inds