from __future__ import division

import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
import matplotlib.pylab as plt
import matplotlib.path as mplpath

from learning2learn.images import shift_image, generate_random_shape

from learning2learn.mpl_textures import (Texture, generate_texture,
                                         get_base_image, add_texture)

def get_train_test_inds(nb_categories, nb_exemplars, nb_samples, nb_test=1):
    """

    :param nb_categories:
    :param nb_exemplars:
    :param nb_shapes:
    :param nb_test:
    :return:
    """
    test_inds = []
    for i in range(nb_categories):
        bottom = i * (nb_exemplars + nb_test)
        top = bottom + nb_test
        test_inds.extend(range(bottom, top))
    # The train inds are the set difference of all inds and test inds
    train_inds = list(set(range(nb_samples)).difference(test_inds))

    return train_inds, test_inds

def generate_colors(nb_colors):
    """
    Function to generate a set of nb_colors colors. They
    are generated such that there is sufficient distance
    between each color vector. This is better than random
    color sampling.
    :param nb_colors: (int) the number of colors to generate
    :return: (Numpy array) the (nb_colors, 3) color matrix
    """
    nb_bins = np.round(np.power(nb_colors, 1 / 3)) + 1
    vals = np.linspace(0, 0.95, int(nb_bins))
    colors = []
    for r in vals:
        for g in vals:
            for b in vals:
                colors.append([r, g, b])

    colors = sorted(colors, key=lambda x: sum(x))
    colors = colors[-nb_colors:]
    return np.asarray(colors)

def build_vocab_training_set(data_folder, nb_exemplars, nb_categories,
                             shape_fraction, color_fraction, shift=True):
    # Load the data
    imgs, df = load_images(data_folder, target_size=(200, 200), shift=shift)
    # Select the classes
    nb_shapes = int(nb_categories * shape_fraction)
    nb_colors = int(nb_categories * color_fraction)
    nb_textures = nb_categories - nb_shapes - nb_colors
    assert nb_shapes <= 50 and nb_colors <= 50 and nb_textures <= 50
    assert (nb_shapes + nb_colors + nb_textures) == 50
    print('Using %i shape words, %i color words and %i texture words.' %
          (nb_shapes, nb_colors, nb_textures))
    shapes = np.random.choice(range(nb_categories), nb_shapes, replace=False)
    colors = np.random.choice(range(nb_categories), nb_colors, replace=False)
    textures = np.random.choice(range(nb_categories), nb_textures,
                                replace=False)
    # ...
    inds = []
    labels = []
    current_class = 0
    for s in shapes:
        ix = np.where(df['shape'].as_matrix() == s)[0]
        inds.extend(list(np.random.choice(ix, nb_exemplars, replace=False)))
        labels.extend([current_class] * nb_exemplars)
        current_class += 1
    for c in colors:
        ix = np.where(df['color'].as_matrix() == c)[0]
        inds.extend(list(np.random.choice(ix, nb_exemplars, replace=False)))
        labels.extend([current_class] * nb_exemplars)
        current_class += 1
    for t in textures:
        ix = np.where(df['texture'].as_matrix() == t)[0]
        inds.extend(list(np.random.choice(ix, nb_exemplars, replace=False)))
        labels.extend([current_class] * nb_exemplars)
        current_class += 1

    return imgs[inds], np.asarray(labels)

def select_subset(df, nb_select):
    """
    Helper function for load_image_dataset. If we are subsampling nb_select
    samples from a particular category, we would like to choose them such that
    the colors are optimally spaced. This function does so.
    :param df:
    :param nb_select:
    :return:
    """
    assert nb_select <= 15, 'only 15 exemplars to select of each category'
    nb_categories = df.shape[0]
    # Sort by color values, get the indices
    ix = df.sort_values(by='color').index
    if nb_select == len(ix):
        return ix.tolist()
    else:
        step = int(np.ceil(nb_categories / nb_select)) - 1
        return [ix[i * step] for i in range(nb_select)]

def load_images(data_folder, target_size=(200, 200), shift=True):
    # First load the images
    files = [file for file in os.listdir(data_folder) if file.endswith('png')]
    files = sorted(files)
    imgs = np.zeros(shape=(len(files),)+target_size+(3,), dtype=np.float32)
    for i, file in enumerate(files):
        img_path = os.path.join(data_folder, file)
        img = image.load_img(img_path, target_size=target_size,
                             interpolation='bicubic')
        img = image.img_to_array(img) / 255.
        if shift:
            img = shift_image(img, img_size=target_size)
        imgs[i] = img
    # Now load the feature info
    feature_file = os.path.join(data_folder, 'data.csv')
    df = pd.read_csv(feature_file, index_col=0)

    return imgs, df

def load_image_dataset(data_folder, nb_categories=None, nb_exemplars=None,
                       nb_test=5, target_size=(200, 200)):
    # First load the data
    imgs, df = load_images(data_folder, target_size)
    if nb_categories is None:
        # if these two parameters are 'None' we will not subsample the data.
        # simply load and return the images.
        assert nb_exemplars is None
        return imgs
    # Collect a subset of the data according to {nb_categories, nb_exemplars}
    ix = []
    for cat in range(nb_categories):
        ix_cat = select_subset(df[df['shape'] == cat], nb_exemplars + nb_test)
        ix_cat = list(np.random.permutation(ix_cat))
        ix.extend(ix_cat)
    imgs = imgs[ix]
    df = df.iloc[ix]

    return imgs, df['shape'].as_matrix()

def generate_dataset_parameters(nb_categories, image_size=500,
                                mpl_textures=False):
    """

    :param nb_categories:
    :param image_size:
    :param mpl_textures:
    :return:
    """
    # Generate shapes, which are sets of points for which polygons will
    # be generated
    shapes = [generate_random_shape(0, 500, 0, 500, 100) for _ in
              range(nb_categories)]
    # Generate colors, which are 3-D vectors of values between 0-1 (RGB values)
    colors = generate_colors(nb_categories)
    if mpl_textures:
        # using matplotlib custom textures
        patch_types = [
                'ellipse', 'arc', 'arrow', 'circle',
                'rectangle', 'wedge', 'pentagon'
            ]
        nb_variations = max(int(np.ceil(nb_categories / len(patch_types))), 3)
        textures = []
        for patch_type in patch_types:
            t_list = [generate_texture(patch_type, image_size=image_size)
                      for _ in range(nb_variations)]
            textures.extend(t_list)
        hatch_types = ['/', '-', '+']
        for hatch_type in hatch_types:
            textures.append(Texture(hatch_type, gradient=None))
            textures.append(Texture(hatch_type, gradient='right'))
            textures.append(Texture(hatch_type, gradient='left'))
        textures = np.random.choice(textures, nb_categories, replace=False)
    else:
        # using pre-designed textures, which are saved as image files in the
        # 'data' subfolder
        assert os.path.isdir('../data/textures')
        files = sorted([file for file in os.listdir('../data/textures') if
                        file.endswith('tiff')])
        assert nb_categories <= len(files)

        # return np.random.choice(files, nb_textures, replace=False)
        textures = files[:nb_categories]

    return shapes, colors, textures

def generate_image(shape, color, texture, save_file, mpl_textures=False):
    """

    :param shape:
    :param color:
    :param texture:
    :param save_file:
    :return:
    """
    # Generate the base image and save it to a file
    if mpl_textures:
        img = get_base_image(500, 500, color, shape, gradient=texture.gradient)
    else:
        img = image.load_img('../data/textures/%s' % texture,
                             target_size=(500, 500))
        img = image.img_to_array(img) / 255.
        # normalize the texture for color consistency. 0.57248
        # is the average activation for the whole texture dataset.
        img *= (0.57248/np.mean(img))
        img = np.minimum(img, 1.)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if mpl_textures:
        ax.imshow(img, interpolation='bicubic')
        add_texture(ax, texture, 500)
    else:
        ax.imshow(img*color, interpolation='bicubic')
    plt.savefig(save_file)
    plt.close()
    # Load the base image from file, crop it using mplpath,
    # and save back to the file
    img = image.load_img(save_file, target_size=(500, 500))
    img = image.img_to_array(img)
    img /= 255.
    p = mplpath.Path(shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not p.contains_point((i, j)):
                img[j,i,:] = np.array([1.,1.,1.])
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()