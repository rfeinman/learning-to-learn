from __future__ import division
import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import keras.backend as K


# def get_train_test_inds(nb_categories, nb_exemplars, nb_samples, nb_test=1):
#     """
#
#     :param nb_categories:
#     :param nb_exemplars:
#     :param nb_shapes:
#     :param nb_test:
#     :return:
#     """
#     test_inds = []
#     for i in range(nb_categories):
#         bottom = i * (nb_exemplars + nb_test)
#         top = bottom + nb_test
#         test_inds.extend(range(bottom, top))
#     # The train inds are the set difference of all inds and test inds
#     train_inds = list(set(range(nb_samples)).difference(test_inds))
#
#     return train_inds, test_inds

def train_test_split(x, test_size):
    step = int(np.ceil(len(x) / test_size)) - 1
    ix = list(range(len(x)))
    ix_test = [i * step for i in range(test_size)]
    ix_train = list(set(ix).difference(ix_test))
    x_train = [x[i] for i in ix_train]
    x_test = [x[i] for i in ix_test]

    return x_train, x_test

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
    # Define the Keras function that will return features
    get_features = K.function([model.layers[0].input, K.learning_phase()],
                              [model.layers[layer_num].output])
    # Now, run through batches and compute features
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    outputs = []
    for i in range(n_batches):
        outputs.append(get_features([X[i*batch_size:(i+1)*batch_size], 0])[0])

    # Concatenate the list of arrays and return
    return np.concatenate(outputs)

def similarity(x1, x2):
    """
    Computes the cosine similarity between two vectors.
    :param x1: (Numpy array) The first vector.
    :param x2: (Numpy array) The second vector.
    :return: (int) The similarity score.
    """

    return 1 - cosine(x1, x2)

def evaluate_secondOrder(model, X, layer_num, batch_size=32):
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
    :param layer_num: (int) The index of the layer whose representation will be
                        used for similarity evaluation
    :param batch_size: (int) The batch size to use when evaluating the model
                        on a set of inputs.
    :return: (float) The fraction of groupings in which the shape constant
                    sample was most similar to the baseline sample.
    """
    # Since we have groupings of 4 samples, X should have a length that is a
    # multiple of 4.
    assert len(X) % 4 == 0
    X_p = get_hidden_representations(model, X, layer_num=layer_num,
                                     batch_size=batch_size)
    nb_correct = 0
    for i in range(int(len(X) / 4)):
        score_shape = similarity(X_p[4*i], X_p[4*i+1])
        score_color = similarity(X_p[4*i], X_p[4*i+2])
        score_texture = similarity(X_p[4*i], X_p[4*i+3])
        if score_shape > score_color and score_shape > score_texture:
            nb_correct += 1

    # Return the percentage of times we were correct
    return nb_correct / float(len(X)/4)

def save_results(cats, exemps, scores, save_path):
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

def experiment_loop(exectue_fn, category_trials, exemplar_trials, params,
                    results_path):
    cats = []
    exemps = []
    scores = []
    stdout = sys.stdout
    # Create results_path folder. Remove previous one if it already exists.
    if os.path.isdir(results_path):
        warnings.warn('Removing old results folder of the same name!')
        shutil.rmtree(results_path)
    os.mkdir(results_path)
    # Loop through different values of (nb_categories, nb_exemplars)
    for nb_categories in category_trials:
        for nb_exemplars in exemplar_trials:
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            log_file = os.path.join(results_path,
                                    'log_ca%0.4i_ex%0.4i' %
                                    (nb_categories, nb_exemplars))
            sys.stdout = open(log_file,'w')
            result = exectue_fn(nb_categories, nb_exemplars, params)
            sys.stdout = stdout
            cats.append(nb_categories)
            exemps.append(nb_exemplars)
            scores.append(result)
            # Save results from this run to text file
            save_file = os.path.join(results_path, 'results.csv')
            save_results(cats, exemps, scores, save_file)
    print('Experiment loop complete.')

def train_model(model, X_train, Y_train, epochs, validation_data,
                batch_size, checkpoint, burn_period=20):
    """
    A helper function for training a Keras model with a 'burn period.'
    The burn period is an initial set of epochs for which the model will
    not be saved, before using the callback ModelCheckpoint. This helps save
    time as model saving is very slow, and we usually end up with a better model
    down the line after these initial epochs anyways.
    :param model:
    :param X_train:
    :param Y_train:
    :param epochs:
    :param validation_data:
    :param batch_size:
    :param checkpoint:
    :param burn_period:
    :return:
    """
    if burn_period < epochs:
        # burn period training
        model.fit(
            X_train, Y_train, epochs=burn_period,
            shuffle=True, validation_data=validation_data,
            verbose=1, batch_size=batch_size
        )
        # training beyond burn period; start saving best model
        model.fit(
            X_train, Y_train, epochs=epochs-burn_period,
            shuffle=True, validation_data=validation_data,
            verbose=1, batch_size=batch_size,
            callbacks=[checkpoint]
        )
    else:
        model.fit(
            X_train, Y_train, epochs=epochs,
            shuffle=True, validation_data=validation_data,
            verbose=1, batch_size=batch_size,
            callbacks=[checkpoint]
        )

# def build_vocab_training_set(data_folder, nb_exemplars, nb_categories,
#                              shape_fraction, color_fraction, shift=True):
#     # Load the data
#     imgs, df = load_images(data_folder, target_size=(200, 200), shift=shift)
#     # Select the classes
#     nb_shapes = int(nb_categories * shape_fraction)
#     nb_colors = int(nb_categories * color_fraction)
#     nb_textures = nb_categories - nb_shapes - nb_colors
#     assert nb_shapes <= 50 and nb_colors <= 50 and nb_textures <= 50
#     assert (nb_shapes + nb_colors + nb_textures) == 50
#     print('Using %i shape words, %i color words and %i texture words.' %
#           (nb_shapes, nb_colors, nb_textures))
#     shapes = np.random.choice(range(nb_categories), nb_shapes, replace=False)
#     colors = np.random.choice(range(nb_categories), nb_colors, replace=False)
#     textures = np.random.choice(range(nb_categories), nb_textures,
#                                 replace=False)
#     # ...
#     inds = []
#     labels = []
#     current_class = 0
#     for s in shapes:
#         ix = np.where(df['shape'].as_matrix() == s)[0]
#         inds.extend(list(np.random.choice(ix, nb_exemplars, replace=False)))
#         labels.extend([current_class] * nb_exemplars)
#         current_class += 1
#     for c in colors:
#         ix = np.where(df['color'].as_matrix() == c)[0]
#         inds.extend(list(np.random.choice(ix, nb_exemplars, replace=False)))
#         labels.extend([current_class] * nb_exemplars)
#         current_class += 1
#     for t in textures:
#         ix = np.where(df['texture'].as_matrix() == t)[0]
#         inds.extend(list(np.random.choice(ix, nb_exemplars, replace=False)))
#         labels.extend([current_class] * nb_exemplars)
#         current_class += 1
#
#     return imgs[inds], np.asarray(labels)