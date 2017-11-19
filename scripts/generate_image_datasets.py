from __future__ import division, print_function
import os
import matplotlib as mpl
mpl.use('Agg')

from learning2learn.images import generate_dataset_parameters, generate_image
from learning2learn.util import synthesize_data

def create_dataset(nb_categories, nb_exemplars, data_folder):
    # Generate the set of shapes, colors and textures that we will draw from
    shape_set, color_set, texture_set = \
        generate_dataset_parameters(nb_categories)
    # Create folder where images will be stored; if exists, remove
    if os.path.isdir(data_folder):
        print('A dataset of the specified parameters already exists. Skipping '
              'this one...')
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

def main():
    for ca in [35]:
        for ex in [1, 2, 3]:
            print('Generating image dataset for %i categories and '
                  '%i exemplars...' % (ca, ex))
            data_folder = os.path.realpath('../data/images_ca%0.4i_ex%0.4i' %
                                           (ca, ex))
            create_dataset(ca, ex, data_folder)

if __name__ == "__main__":
    main()