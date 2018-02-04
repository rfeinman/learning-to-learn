from __future__ import division
import os
import sys
import shutil
import warnings
import numpy as np
from scipy.spatial.distance import cosine
import keras.backend as K


def similarity(x1, x2, measure='cosine'):
    assert measure in ['cosine', 'euclidean']
    if measure == 'cosine':
        return 1 - cosine(x1, x2)
    else:
        return -np.linalg.norm(x1 - x2)

def train_test_split(x, test_size):
    assert type(test_size) == int, 'test_size parameter must be an int'
    if test_size <= (len(x)/2.):
        nb_sample = test_size
    else:
        nb_sample = len(x) - test_size
    # evenly space selections accross the range of the input list
    step = int(np.floor(len(x) / nb_sample))
    ix = list(range(len(x)))
    ix1 = [i * step for i in range(nb_sample)]
    # center the selections
    diff = len(x) - 1 - max(ix1)
    shift = int(np.floor(diff/2.))
    ix1 = [i+shift for i in ix1]
    # the test set is the converse of the train set
    ix2 = list(set(ix).difference(ix1))
    if test_size <= (len(x)/2.):
        x_train = [x[i] for i in ix2]
        x_test = [x[i] for i in ix1]
    else:
        x_train = [x[i] for i in ix1]
        x_test = [x[i] for i in ix2]

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

def evaluate_generalization(model, X, layer_num, batch_size=32):
    """
    Evaluate a trained Keras model on a set of novel objects. The novel objects
    come in groupings of 4, where each grouping contains an exemplar sample, a
    shape-match sample, a color-match sample, and a texture-match
    sample. For each grouping, we find which of the 3 match samples is most
    similar to the exemplar sample according to the model's internal features.
    Then, we compute the fraction of times that it was the correct (shape
    match) sample.
    :param model: (Keras Sequential) The Keras model to be used for evaluation.
    :param X: (Numpy array) The input data.
    :param layer_num: (int) The index of the layer whose representation will be
                        used for similarity evaluation
    :param batch_size: (int) The batch size to use when evaluating the model
                        on a set of inputs.
    :return: (tuple) For each of the 3 match types, the fraction of groupings in
                    which the sample of the particular match was selected.
    """
    # Since we have groupings of 4 samples, X should have a length that is a
    # multiple of 4.
    assert len(X) % 4 == 0
    X_p = get_hidden_representations(model, X, layer_num=layer_num,
                                     batch_size=batch_size)
    # keep a count of the # times each match is selected (indexed 0,1,3)
    counts = {0:0, 1:0, 2:0}
    for i in range(int(len(X) / 4)):
        scores = np.zeros(3)
        scores[0] = similarity(X_p[4*i], X_p[4*i+1]) # shape match score
        scores[1] = similarity(X_p[4*i], X_p[4*i+2]) # color match score
        scores[2] = similarity(X_p[4*i], X_p[4*i+3]) # texture match score
        # Find location of max
        ix_best = np.where(scores == np.max(scores))[0]
        # If multiple scores were max, then we want to random sample from them
        if len(ix_best) > 1:
            match = np.random.choice(ix_best, 1)[0]
        else:
            match = ix_best[0]
        counts[match] += 1

    # Return the percentages for each of the 3 match types
    return counts[0]/(len(X)/4.), counts[1]/(len(X)/4.), counts[2]/(len(X)/4.)

def experiment_loop(
        exectue_fn, category_trials, exemplar_trials, params, results_path
):
    stdout = sys.stdout
    # Create results_path folder. Remove previous one if it already exists.
    if os.path.isdir(results_path):
        warnings.warn('Removing old results folder of the same name!')
        shutil.rmtree(results_path)
    os.mkdir(results_path)
    np.save(os.path.join(results_path, 'category_trials.npy'),
            np.asarray(category_trials))
    np.save(os.path.join(results_path, 'exemplar_trials.npy'),
            np.asarray(exemplar_trials))
    # Loop through different values of (nb_categories, nb_exemplars)
    results_O1 = np.zeros(
        shape=(len(category_trials), len(exemplar_trials), params['nb_runs'])
    )
    results_O2 = np.zeros(
        shape=(len(category_trials), len(exemplar_trials), params['nb_runs'])
    )
    for i, nb_categories in enumerate(category_trials):
        for j, nb_exemplars in enumerate(exemplar_trials):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            log_file = os.path.join(results_path,
                                    'log_ca%0.4i_ex%0.4i' %
                                    (nb_categories, nb_exemplars))
            sys.stdout = open(log_file,'w')
            results_O1[i, j], results_O2[i, j] = \
                exectue_fn(nb_categories, nb_exemplars, params)
            sys.stdout = stdout
            # Save results from this run to text file
            save_file_gen = os.path.join(results_path, 'results_O1.npy')
            np.save(save_file_gen, results_O1)
            save_file_gen = os.path.join(results_path, 'results_O2.npy')
            np.save(save_file_gen, results_O2)
    print('Experiment loop complete.')

def train_model(
        model, X_train, Y_train, epochs, validation_data, batch_size,
        checkpoint, burn_period=20
):
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