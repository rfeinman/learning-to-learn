from __future__ import division
import os
import argparse
import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.models import simple_cnn
from learning2learn.images import generate_dataset_parameters, generate_image
from learning2learn.util import synthesize_data, save_results

def create_dataset(nb_categories, nb_exemplars, data_folder):
    # Generate the set of shapes, colors and textures that we will draw from
    shape_set, color_set, texture_set = \
        generate_dataset_parameters(nb_categories)
    # Create folder where images will be stored; if exists, remove
    if os.path.isdir(data_folder):
        print('A dataset of the specified parameters already exists. Using '
              'the existing one...')
    else:
        print('Building the image dataset...')
        os.mkdir(data_folder)
        # Synthesize the dataset. Use nb_exemplars+1 because 1 exemplar
        # of each class will be used for testing.
        df, _ = synthesize_data(nb_categories, nb_exemplars+1)
        shapes = [shape_set[i] for i in df['shape']]
        colors = [color_set[i] for i in df['color']]
        textures = [texture_set[i] for i in df['texture']]
        for shape, color, texture, i in \
                zip(shapes, colors, textures, range(len(shapes))):
            save_file = os.path.join(data_folder, 'img%0.4i.png' % i)
            generate_image(shape, color, texture, save_file)
        # Save the dataset parameters so we know what we're working with
        df.to_csv(os.path.join(data_folder, 'data.csv'))

def load_dataset(data_folder):
    # First load the images
    imgs = []
    files = [file for file in os.listdir(data_folder) if file.endswith('png')]
    files = sorted(files)
    for file in files:
        img_path = os.path.join(data_folder, file)
        img = image.load_img(img_path, target_size=(200, 200))
        imgs.append(image.img_to_array(img))
    imgs = np.asarray(imgs)
    imgs /= 255.
    # Now load the feature info
    feature_file = os.path.join(data_folder, 'data.csv')
    df = pd.read_csv(feature_file, index_col=0)
    shapes = df['shape'].as_matrix()

    return imgs, shapes

def run_experiment(nb_categories, nb_exemplars):

    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    data_folder = os.path.realpath('../data/images_ca%0.4i_ex%0.4i' %
                                   (nb_categories, nb_exemplars))
    create_dataset(nb_categories, nb_exemplars, data_folder)
    X, shapes = load_dataset(data_folder)
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(shapes.reshape(-1, 1))
    # Now, we separate the train and test sets
    test_inds = [i*(nb_exemplars+1) for i in range(nb_categories)]
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(len(shapes))).difference(test_inds))
    # Build a neural network model and train it with the training set
    print('Training CNN model...')
    model = simple_cnn(input_shape=X.shape[1:], nb_classes=Y.shape[-1])
    model.fit(X[train_inds], Y[train_inds], epochs=args.nb_epochs,
              shuffle=True, validation_data=(X[test_inds], Y[test_inds]),
              verbose=0)
    loss, acc = model.evaluate(X[test_inds], Y[test_inds], verbose=0)

    return acc

def main():
    """
    The main script code.
    :param args: (Namespace object) Command line arguments.
    """
    cats = []
    exemps = []
    scores = []
    # Loop through different values of (nb_categories, nb_exemplars)
    for nb_categories in range(5, 51, 5):
        for nb_exemplars in range(1, 15):
            print('Testing for %i categories and %i exemplars...' %
                  (nb_categories, nb_exemplars))
            result = run_experiment(nb_categories, nb_exemplars)
            cats.append(nb_categories)
            exemps.append(nb_exemplars)
            scores.append(result)
            # Save results from this run to text file
            save_results(cats, exemps, scores, args.save_path)
    print('Experiment loop complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--nb_epochs',
                        help='The number of epochs to train for.',
                        required=False, type=int)
    parser.add_argument('-sp', '--save_path',
                        help='The file path where results should be saved',
                        required=False, type=str)
    parser.set_defaults(nb_epochs=100)
    parser.set_defaults(save_path='../results/results_firstOrder_image.csv')
    args = parser.parse_args()
    main()