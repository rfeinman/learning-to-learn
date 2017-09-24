from __future__ import division, print_function
import numpy as np
import functools
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def rearrange_points(points):
    """
    A function to sort a list of points in clockwise ordering. This
    will help to ensure that our polygon shapes are whole.
    :param points:
    :return:
    """
    center_x = np.mean([elt[0] for elt in points])
    center_y = np.mean([elt[1] for elt in points])

    # function to compare two points
    def less(a, b):
        # preliminary checks
        if a[0] >= center_x and b[0] < center_x:
            return 1
        if a[0] < center_x and b[0] >= center_x:
            return -1
        if a[0] == center_x and b[0] == center_x:
            if a[1] >= center_y or b[1] >= center_y:
                if a[1] > b[1]:
                    return 1
                else:
                    return -1
            else:
                if b[1] > a[1]:
                    return 1
                else:
                    return -1

        # compute the cross product of vectors (center -> a) x (center -> b)
        det = (a[0] - center_x) * (b[1] - center_y) - \
              (b[0] - center_x) * (a[1] - center_y)
        if det < 0:
            return 1
        elif det > 0:
            return -1

        # Points a and b are on the same line from the center.
        # Check which point is closer to the center.
        d1 = (a[0] - center_x) * (a[0] - center_x) + \
             (a[1] - center_y) * (a[1] - center_y)
        d2 = (b[0] - center_x) * (b[0] - center_x) + \
             (b[1] - center_y) * (b[1] - center_y)
        if d1 > d2:
            return 1
        else:
            return -1

    return sorted(points, key=functools.cmp_to_key(less))

def generate_random_shape(x_min, x_max, y_min, y_max, x_offset, y_offset):
    """

    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param x_offset:
    :param y_offset:
    :return:
    """
    x = np.random.uniform(x_min, x_max - x_offset)
    y = np.random.uniform(y_min, y_max - y_offset)
    points = [(np.random.uniform(x, x + x_offset),
               np.random.uniform(y, y + y_offset)) for _ in
              range(randint(3, 10))]
    points = rearrange_points(points)
    return points

def generate_dataset_parameters(nb_shapes):
    """

    :param nb_shapes:
    :return:
    """
    # Generate shapes, which are sets of points for which polygons will
    # be generated
    shapes = [generate_random_shape(-2, 2, -2, 2, 4, 4) for _ in
              range(nb_shapes)]
    # Generate colors, which are 3-D vectors of values between 0-1 (RGB values)
    colors = np.random.uniform(0, 1, size=(nb_shapes, 3))
    # Generate textures
    texture_set = ['-', '+', 'x', '/', '*', 'o', 'O', '.']
    nb_variations = int(np.ceil(nb_shapes / len(texture_set)))
    textures = []
    for texture in texture_set:
        textures.extend([i * texture for i in range(1, nb_variations + 1)])
    textures = np.random.choice(textures, nb_shapes, replace=False)

    return shapes, colors, textures

def generate_image(shape, color, texture, save_file):
    """

    :param shape:
    :param color:
    :param texture:
    :param save_file:
    :return:
    """
    # Plot figure
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    # Lay the baseline color
    polygon = Polygon(shape, closed=True, fill=True, color=color)
    obj = ax.add_patch(polygon)
    # Lay the texture hatch
    polygon_h = Polygon(shape, closed=True, fill=False, color=(0, 0, 0))
    obj_h = ax.add_patch(polygon_h)
    obj_h.set_hatch(texture)
    plt.ylim([-2,2])
    plt.xlim([-2,2])
    plt.axis('off')
    fig.savefig(save_file)

def resize(img_array, width, height, color=True):
    """
    TODO
    :param img_array:
    :param width:
    :param height:
    :param color:
    :return:
    """
    if color:
        nb_channels = 3
    else:
        nb_channels = 1
    # convert to PIL image
    img = Image.fromarray(img_array)
    # resize using PIL
    img = img.resize((width, height), PIL.Image.ANTIALIAS)
    # convert back to numpy array and return
    return 1 - np.array(img.getdata()).reshape(img.size[0], img.size[1], nb_channels)