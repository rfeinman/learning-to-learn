from __future__ import division
import os
import random
import numpy as np
import functools
from keras.preprocessing import image
import matplotlib.path as mplpath

from . import tex


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

def generate_random_shape(x_min, x_max, y_min, y_max, edge_distance):
    # Sample a number of points for the polygon
    nb_points = np.random.randint(3, 11)
    # 4 'types' of points; determines the edge that the point will be near
    point_types = ['left', 'right', 'top', 'bottom']
    # Cycle through drawing points of different types
    points = []
    for i in range(nb_points):
        if point_types[i % 4] in ['left', 'right']:
            x = np.random.uniform(0, edge_distance)
            y = np.clip(
                np.random.normal(loc=(y_max - y_min) / 2,
                                 scale=(y_max - y_min) / 8),
                y_min,
                y_max
            )
            if point_types[i % 4] == 'right':
                x = x_max - x
        elif point_types[i % 4] in ['top', 'bottom']:
            x = np.clip(
                np.random.normal(loc=(x_max - x_min) / 2,
                                 scale=(x_max - x_min) / 8),
                x_min,
                x_max
            )
            y = np.random.uniform(0, edge_distance)
            if point_types[i % 4] == 'bottom':
                y = y_max - y
        points.append((x, y))
    # Rearrange the points so that they are in the correct order
    points = rearrange_points(points)
    ## Now center the points by computing the mean distance
    ## from the center and then subtracting this mean
    # x_mean = np.mean([p[0] - (x_max-x_min)/2 for p in points])
    # y_mean = np.mean([p[1] - (y_max-y_min)/2 for p in points])
    # points = [(p[0]-x_mean, p[1]-y_mean) for p in points]

    return points

def generate_colors():
    nb_colors = 64
    nb_bins = 4
    vals = np.linspace(0, 0.9, nb_bins)
    colors = np.zeros(shape=(nb_colors, 3))
    i = 0
    for r in vals:
        for g in vals:
            for b in vals:
                colors[i] = np.asarray([r, g, b])
                i += 1

    return colors

def compute_area(shape, img_size=200):
    area = 0
    p = mplpath.Path(shape)
    for i in range(img_size):
        for j in range(img_size):
            if p.contains_point((i, j)):
                area += 1

    return area

def shift_image(img, img_size=(200, 200), scale=20):
    # compute shape boundaries
    y_min = min(np.where(img < 1.)[0])
    y_max = max(np.where(img < 1.)[0])
    x_min = min(np.where(img < 1.)[1])
    x_max = max(np.where(img < 1.)[1])
    # randomly select offsets from a uniform R.V. The boundaries
    # are set such that we don't cut off the object.
    ox = np.random.randint(low=max(-scale, -x_min),
                           high=min(scale, img_size[0] - x_max))
    oy = np.random.randint(low=max(-scale, -y_min),
                           high=min(scale, img_size[1] - y_max))
    # shift the image by offsets
    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0, s)
    shift_img = np.ones_like(img, dtype=np.float32)
    shift_img[mom(oy):non(oy), mom(ox):non(ox)] = img[mom(-oy):non(-oy),
                                                  mom(-ox):non(-ox)]

    return shift_img

def shift_images(imgs, shift_scale=20):
    img_size = imgs.shape[1:3]
    imgs_p = imgs
    for i in range(len(imgs)):
        imgs_p[i, :, :, :3] = shift_image(
            imgs[i, :, :, :3], img_size=img_size, scale=shift_scale
        )

    return imgs_p

def generate_image(shape, color, texture, target_size=(200, 200),
                   shift_scale=20):
    assert shift_scale >= 0 and type(shift_scale) == int
    # Generate the base color
    img_color = np.ones(shape=target_size + (3,), dtype=np.float32) * color
    # Generate the base texture
    img_texture = image.load_img(
        os.path.join(os.path.dirname(tex.__file__), texture),
        target_size=target_size,
        interpolation='bicubic'
    )
    img_texture = image.img_to_array(img_texture) / 255.
    img_texture = img_texture[:, :, 0]
    # Put it all together
    img = np.ones(shape=target_size + (4,), dtype=np.float32)
    img[:, :, :3] = img_color
    # img[:,:,3] = img_texture
    # Cutout the shape
    p = mplpath.Path(shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not p.contains_point((i, j)):
                img[j, i, :] = np.ones_like(img[j, i])
    if shift_scale > 0:
        img = shift_image(img, img_size=target_size, scale=shift_scale)
    img[:, :, 3] = img_texture

    return img

def generate_image_wrapper(tup):
    # since images are randomly shifted, we want a different random seed
    # for each process
    seed = random.randint(0, 1e7)
    np.random.seed(seed)
    return generate_image(tup[0], tup[1], tup[2], tup[3], tup[4])