from __future__ import division
import itertools
import warnings
import math
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras.backend as K


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
    """

    :param nb_categories:
    :param nb_exemplars:
    :return:
    """
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

def get_hidden_representations(model, X, layer_num, batch_size=32):
    """
    Given a Keras model and a data matrix X, this function computes the hidden
    layer representations of the model for each sample in X. The user must
    specify which layer is desired.
    :param model: (Keras Sequential) The model to use for translation
    :param X: (Numpy array) The input data.
    :param layer_num: (int) The layer number we'd like to use as output.
    :param batch_size: (int) The batch size to use when evaluating the model
                        on a set of inputs.
    :return: (Numpy array) The output features.
    """
    # Record the hidden layer dimensionality
    output_dim = model.layers[layer_num].output.shape[-1].value
    # Define the Keras function that will return features
    get_features = K.function([model.layers[0].input],
                              [model.layers[layer_num].output])
    # Now, run through batches and compute features
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i*batch_size:(i+1)*batch_size] = \
            get_features([X[i*batch_size:(i+1)*batch_size], 0])[0]

    return output

def similarity(x1, x2):
    """
    Computes the cosine similarity between two vectors.
    :param x1: (Numpy array) The first vector.
    :param x2: (Numpy array) The second vector.
    :return: (int) The similarity score.
    """
    return 1 - cosine(x1, x2)

def evaluate_secondOrder(model, X, batch_size=32):
    """
    Evaluate a trained Keras model on a set of novel objects. The novel objects
    come in groupings of 4, where each grouping contains a baseline sample, a
    shape constant sample, a color constant sample, and a texture constant
    sample. For each grouping, we find which of the other 3 samples is most
    similar to the baseline sample according to the model's internal features.
    Then, we compute the fraction of times that it was the correct (shape
    constant) sample.
    :param model: (Keras Sequential) The Keras model to be used for evaluation.
    :param X: (Numpy array) The input data.
    :param batch_size: (int) The batch size to use when evaluating the model
                        on a set of inputs.
    :return: (float) The fraction of groupings in which the shape constant
                    sample was most similar to the baseline sample.
    """
    # Since we have groupings of 4 samples, X should have a length that is a
    # multiple of 4.
    assert len(X) % 4 == 0
    X_p = get_hidden_representations(model, X, layer_num=1,
                                     batch_size=batch_size)
    nb_correct = 0
    for i in range(int(len(X) / 4)):
        score_shape = similarity(X_p[4*i], X_p[4*i+1])
        score_color = similarity(X_p[4*i], X_p[4*i+2])
        score_texture = similarity(X_p[4*i], X_p[4*i+3])
        if np.argmax([score_shape, score_color, score_texture]) == 0:
            nb_correct += 1

    # Return the percentage of times we were correct
    return nb_correct / float(len(X)/4)

def save_results(cats, exemps, scores, save_path):
    """
    TODO
    :param cats:
    :param exemps:
    :param scores:
    :param save_path:
    :return:
    """
    df = pd.DataFrame(index=np.unique(exemps), columns=np.unique(cats))
    for c, e, s in zip(cats, exemps, scores):
        df[c].loc[e] = s
    df.to_csv(save_path, index=True)

def add_noise(X, p):
    """
    Randomly flip the bits of the input binary feature matrix X with some
    probability p.
    :param X: (Numpy array) The input feature matrix. Must contain 0/1s only
    :param p: (float) A value between 0-1 that represents the probability
                of a given bit flipping.
    :return: (Numpy array) The corrupted feature matrix
    """
    mask = np.random.binomial(n=1, p=p, size=X.shape)
    return np.abs(X - mask)