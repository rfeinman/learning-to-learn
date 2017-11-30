from __future__ import division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder

from vgg_layer_biases import run_experiment, plot_vgg_results


def load_image_dataset(save_folder, target_size):
    shapes = []
    colors = []
    textures = []
    imgs = []
    files = [file for file in os.listdir(save_folder) if
             file.endswith('jpg') or file.endswith('png')]
    for file in files:
        shape, texture, color = file.split('_')
        color = color[:-4]
        shapes.append(shape)
        textures.append(texture)
        colors.append(color)
        img_path = os.path.join(save_folder, file)
        img = image.load_img(img_path, target_size=target_size)
        imgs.append(image.img_to_array(img))

    le = LabelEncoder()
    shapes = le.fit_transform(shapes)
    colors = le.fit_transform(colors)
    textures = le.fit_transform(textures)

    return preprocess_input(np.asarray(imgs)), shapes, colors, textures

def main():
    if args.gpu_num is not None:
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=args.gpu_num)
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        K.set_session(sess)
    data_folder = os.path.realpath('../data/images_artist')
    model = VGG16(weights='imagenet', include_top=True)
    X, shapes, colors, textures = load_image_dataset(
        data_folder, target_size=(224, 224)
    )
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
    save_path = os.path.realpath('../results/vgg_artist.png')
    plot_vgg_results(shape_vals_1, color_vals_1, texture_vals_1, layers_1, save_path)
    if args.gpu_num is not None:
        K.clear_session()
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.set_defaults(gpu_num=None)
    parser.set_defaults(batch_size=32)
    args = parser.parse_args()
    main()