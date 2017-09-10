import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from toy_neuralnet.models import simple_mlp
from toy_neuralnet.util import synthesize_data, preprocess_data
import keras.backend as K


def synthesize_new_data(nb_categories, nb_textures, nb_colors):
    """
    Synthesize new object data for the second order generalization test.
    Groupings of 4 samples are generated: first, a baseline example of the new
    object category, and then 3 comparison samples. One of the comparison
    samples maintains the same shape as the baseline, another the same
    color, and another the same texture. For each, the other features are
    different from the baseline.
    :param nb_categories: (int) The number of categories in our original data
                            set
    :param nb_textures: (int) The number of textures in our original data set
    :param nb_colors: (int) The number of textures in our original data set
    :return: (Pandas DataFrame, Pandas Series) The data features and labels
    """
    # Create the first grouping
    a = np.asarray([[nb_categories, nb_colors, nb_textures],
                    [nb_categories, nb_colors+1, nb_textures+1],
                    [nb_categories+1, nb_colors, nb_textures+2],
                    [nb_categories+2, nb_colors+2, nb_textures]])
    # Loop through, incrementing grouping by 3 each time and stacking them all
    # on top of each other
    dfs = []
    labels = []
    for i in range(nb_categories):
        dfs.append(pd.DataFrame(a+3*i, columns=['shape', 'color', 'texture']))
        labels.extend(['obj%0.8i' % (nb_categories+3*i) for j in range(4)])
    return pd.concat(dfs), pd.Series(labels)

def get_hidden_representations(model, X, layer_num, batch_size=32):
    """
    Takes
    :param model:
    :param X:
    :param layer_num:
    :param batch_size:
    :return:
    """
    # record the hidden layer dimensionality
    output_dim = model.layers[layer_num].output.shape[-1].value
    # define the Keras function that will return features
    get_features = K.function([model.layers[0].input],
                              [model.layers[layer_num].output])
    # now, run through batches and compute features
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i*batch_size:(i+1)*batch_size] = \
            get_features([X[i*batch_size:(i+1)*batch_size], 0])[0]

    return output

def similarity(x1, x2):
    """
    TODO
    :param x1:
    :param x2:
    :return:
    """
    return 1 - cosine(x1, x2)

def evaluate_secondOrder(model, X, batch_size=32):
    # Since we have groupings of 4 samples, X should have a length that is a
    # multiple of 4.
    assert len(X) % 4 == 0
    X_p = get_hidden_representations(model, X, layer_num=1,
                                     batch_size=batch_size)
    nb_correct = 0
    for i in range(len(X) / 4):
        score_shape = similarity(X_p[4*i], X_p[4*i+1])
        score_color = similarity(X_p[4*i], X_p[4*i+2])
        score_texture = similarity(X_p[4*i], X_p[4*i+3])
        if np.argmax([score_shape, score_color, score_texture]) == 0:
            nb_correct += 1

    # return the percentage of times we were correct
    return nb_correct / float(len(X)/4)

def main(args):
    """
    The main script code.
    :param args: (Namespace object) command line arguments
    """
    # Synthesize the training data.
    df, labels = synthesize_data(args.nb_categories,
                                 args.nb_exemplars,
                                 args.nb_textures,
                                 args.nb_colors)
    # Now we will create the test data set for the second-order generalization
    df_new, labels_new = synthesize_new_data(args.nb_categories,
                                             args.nb_textures,
                                             args.nb_colors)
    # Concatenate and pre-process the data
    X, Y = preprocess_data(pd.concat([df, df_new]),
                           pd.concat([labels, labels_new]),
                           one_hot=False, nb_bits=20)
    # Keep track of location of train and test sets
    train_inds = range(df.shape[0])
    test_inds = range(df.shape[0], X.shape[0])
    # Build a neural network model and train it with the training set
    model = simple_mlp(nb_in=X.shape[-1], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, batch_size=32)
    score = evaluate_secondOrder(model, X[test_inds], batch_size=32)
    print('Score: %0.4f' % score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-te', '--nb_textures',
                        help='The number of textures.',
                        required=False, type=int)
    parser.add_argument('-co', '--nb_colors',
                        help='The number of colors.',
                        required=False, type=int)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(nb_categories=100)
    parser.set_defaults(nb_exemplars=5)
    parser.set_defaults(nb_textures=200)
    parser.set_defaults(nb_colors=200)
    args = parser.parse_args()
    main(args)