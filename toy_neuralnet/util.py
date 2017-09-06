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
    """
    TODO
    :param column:
    :param nb_bits:
    :return:
    """
    dictionary = generate_dictionary(column, nb_bits)
    return np.asarray([dictionary[elt] for elt in column])

def preprocess_data(df, labels, one_hot=False, nb_bits=None):
    """
    TODO
    :param df:
    :param labels:
    :param one_hot:
    :param nb_bits:
    :return:
    """
    if one_hot and nb_bits is not None:
        warnings.warn('nb_bits parameter is not used when one_hot=True.')
    if one_hot:
        # one-hot-encode the categorical variables in the dataset
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