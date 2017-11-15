from __future__ import division, print_function
import os
import shutil
import argparse
import warnings
from learning2learn.images import generate_dataset_parameters, generate_image
from learning2learn.util import synthesize_data

def main():
    # Generate the set of shapes, colors and textures that we will draw from
    shape_set, color_set, texture_set = \
        generate_dataset_parameters(args.nb_categories)
    # Create folder where images will be stored; if exists, remove
    if os.path.isdir(args.save_folder):
        warnings.warn('Removing existing folder at %s' % args.save_folder)
        shutil.rmtree(args.save_folder)
    os.mkdir(args.save_folder)
    # Synthesize the dataset
    df, labels = synthesize_data(args.nb_categories, args.nb_exemplars)
    shapes = [shape_set[i] for i in df['shape']]
    colors = [color_set[i] for i in df['color']]
    textures = [texture_set[i] for i in df['texture']]
    for shape, color, texture, i in \
            zip(shapes, colors, textures, range(len(shapes))):
        save_file = os.path.join(args.save_folder, 'img%0.4i.png' % i)
        generate_image(shape, color, texture, save_file)
    # Save the dataset parameters so we know what we're working with
    df.to_csv(os.path.join(args.save_folder, 'data.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--nb_categories',
                        help='The number of categories in the dataset.',
                        required=False, type=int)
    parser.add_argument('-e', '--nb_exemplars',
                        help='The number of exemplars of each category in'
                             'the dataset.',
                        required=False, type=int)
    parser.add_argument('-s', '--save_folder',
                        help='The folder where ...',
                        required=False, type=str)
    parser.set_defaults(nb_categories=10)
    parser.set_defaults(nb_exemplars=2)
    parser.set_defaults(save_folder='../data/image_dataset')
    args = parser.parse_args()
    main()