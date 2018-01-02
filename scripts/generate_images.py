from __future__ import division, print_function
import os
import shutil
import argparse
import warnings
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.images import generate_dataset_parameters, generate_image
from learning2learn.util import synthesize_data

def create_dataset(nb_categories, nb_exemplars, data_folder,
                   mpl_textures):
    # Generate the set of shapes, colors and textures that we will draw from
    shape_set, color_set, texture_set = \
        generate_dataset_parameters(nb_categories, mpl_textures=mpl_textures)
    # Create folder where images will be stored; if exists, remove
    if os.path.isdir(data_folder):
        warnings.warn('A dataset already exists at the specified location...'
                      'overwriting.')
        shutil.rmtree(data_folder)
    print('Building the image dataset...')
    os.mkdir(data_folder)
    # Synthesize the dataset.
    df, _ = synthesize_data(nb_categories, nb_exemplars)
    shapes = [shape_set[i] for i in df['shape']]
    colors = [color_set[i] for i in df['color']]
    textures = [texture_set[i] for i in df['texture']]
    for shape, color, texture, i in \
        zip(shapes, colors, textures, range(len(shapes))):
        save_file = os.path.join(
            data_folder,
            'shape%0.2i_color%0.2i_texture%0.2i.png' %
            (df['shape'].loc[i], df['color'].loc[i], df['texture'].loc[i])
        )
        generate_image(shape, color, texture, save_file, mpl_textures)
    # Save the dataset parameters so we know what we're working with
    df.to_csv(os.path.join(data_folder, 'data.csv'))



def main():
    data_folder = os.path.realpath(args.save_path)
    create_dataset(args.nb_categories, args.nb_exemplars, data_folder,
                   args.mpl_textures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ca', '--nb_categories',
                        help='The number of categories.',
                        required=False, type=int)
    parser.add_argument('-ex', '--nb_exemplars',
                        help='The number of exemplars.',
                        required=False, type=int)
    parser.add_argument('-sp', '--save_path',
                        help='The file path where results should be saved',
                        required=True, type=str)
    parser.add_argument('--mpl_textures', action='store_true', default=False)
    parser.set_defaults(nb_categories=10)
    parser.set_defaults(nb_exemplars=5)
    args = parser.parse_args()
    main()