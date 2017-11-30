from __future__ import division, print_function
import os
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from itertools import combinations
from scipy.spatial.distance import cosine
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from learning2learn.util import get_hidden_representations

def similarity(tuple):
    x1, x2 = tuple
    return 1 - cosine(x1, x2)

def load_and_preprocess_imgs(save_folder, target_size):
    imgs = []
    files = [file for file in os.listdir(save_folder) if file.endswith('png')]
    files = sorted(files)
    for file in files:
        img_path = os.path.join(save_folder, file)
        img = image.load_img(img_path, target_size=target_size,
                             interpolation='bicubic')
        imgs.append(image.img_to_array(img))

    return preprocess_input(np.asarray(imgs))

def accuracy_for_layer(model, X, shape, color, texture, layer_num, batch_size,
                       normalize):
    # Compute hidden representations of the input for a specific
    # hidden layer
    X_h = get_hidden_representations(model, X, layer_num, batch_size)
    # Flatten X_h
    X_h = X_h.reshape(len(X_h), -1)
    # Loop through the samples and compute accuracy
    similarities = []
    for y in [shape, color, texture]:
        means_same = []
        for cat in set(y):
            # Find indices of samples with this category
            inds = list(np.where(y == cat)[0])
            pairs = [(X_h[i], X_h[j]) for i,j in combinations(inds, 2)]
            # with mp.Pool(22) as p:
            #     scores = p.map(similarity, pairs)
            scores = list(map(similarity, pairs))
            means_same.append(np.mean(scores))
        similarities.append(np.mean(means_same))

    # normalize the 3 similarities
    similarities = np.asarray(similarities)
    if normalize:
        similarities /= np.sum(similarities)
    return similarities

def run_experiment(model, X, shapes, colors, textures, batch_size,
                   normalize, verbose=True):
    shape_vals = []
    color_vals = []
    texture_vals = []
    for i in range(len(model.layers)):
        sim_shape, sim_color, sim_texture = accuracy_for_layer(
            model, X, shapes,colors, textures, i, batch_size, normalize
        )
        if verbose:
            print(
                'Results for layer #%i:\n\t shape: %0.3f, color: %0.3f, '
                'texture: %0.3f' %
                (i, sim_shape, sim_color, sim_texture)
            )
        shape_vals.append(sim_shape)
        color_vals.append(sim_color)
        texture_vals.append(sim_texture)

    return shape_vals, color_vals, texture_vals

def plot_vgg_results(shape_vals, color_vals, texture_vals, layers, save_path):
    x_grid = list(range(len(layers)))
    plt.figure(figsize=(10, 8))
    plt.title('VGG-net Biases vs. Layer', fontsize=22)
    plt.plot(x_grid, shape_vals, label='Shape', color='blue')
    plt.plot(x_grid, color_vals, label='Color', color='orange')
    plt.plot(x_grid, texture_vals, label='Texture', color='green')
    plt.yticks(fontsize=14)
    plt.ylabel('Attention strength', fontsize=22)
    plt.xticks(x_grid, layers, rotation='vertical', fontsize=14)
    plt.xlabel('Layer', fontsize=22)
    plt.legend(prop={'size': 15}, loc=3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        K.set_session(sess)
    data_folder = os.path.realpath('../data/images_ca%0.4i_ex%0.4i' %
                                   (args.nb_categories, args.nb_exemplars))
    model = VGG16(weights='imagenet', include_top=True)
    X = load_and_preprocess_imgs(data_folder, target_size=(224, 224))
    data = pd.read_csv(os.path.join(data_folder, 'data.csv'), index_col=0)
    shapes = data['shape'].as_matrix()
    colors = data['color'].as_matrix()
    textures = data['texture'].as_matrix()
    shape_vals, color_vals, texture_vals = run_experiment(
        model, X, shapes, colors, textures, args.batch_size, args.normalize
    )
    layers = ['Input', 'Conv1', 'Conv2', 'MaxPool1', 'Conv3', 'Conv4',
              'MaxPool2', 'Conv5', 'Conv6', 'Conv7', 'MaxPool3', 'Conv8',
              'Conv9', 'Conv10', 'MaxPool4', 'Conv11', 'Conv12', 'Conv13',
              'MaxPool5', 'Flatten', 'Dense1', 'Dense2', 'Softmax']
    inds_keep = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 21]
    layers_1 = [layers[i] for i in inds_keep]
    shape_vals_1 = [shape_vals[i] for i in inds_keep]
    color_vals_1 = [color_vals[i] for i in inds_keep]
    texture_vals_1 = [texture_vals[i] for i in inds_keep]
    save_path = os.path.realpath('../results/vgg_ca%0.4i_ex%0.4i.png' %
                                 (args.nb_categories, args.nb_exemplars))
    plot_vgg_results(shape_vals_1, color_vals_1, texture_vals_1, layers_1, save_path)
    if args.gpu_num is not None:
        K.clear_session()
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-g', '--gpu_num',
                        help='Int indicating which GPU to use',
                        required=False, type=str)
    parser.add_argument('-b', '--batch_size',
                        help='Int indicating the batch size to use',
                        required=False, type=int)
    parser.add_argument('-n', '--normalize',
                        help='Flag indicating that attention scores should be '
                             'normalized to sum to 1',
                        action='store_true')
    parser.set_defaults(nb_categories=10)
    parser.set_defaults(nb_exemplars=10)
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()