from __future__ import absolute_import, division

import numpy as np
from random import randint

from matplotlib.patches import (Arc, Arrow, Circle, CirclePolygon,
                                Ellipse, Rectangle, Wedge, Polygon)

class Texture:
    """
    TODO
    """
    def __init__(self, patch_type, gradient, step=None, params=None):
        """

        :param patch_type:
        :param gradient:
        :param step:
        :param params:
        """
        assert patch_type in [
            'ellipse', 'arc', 'arrow', 'circle',
            'rectangle', 'wedge', 'pentagon',
            '/', '//', '-', '--', '+', '++'
        ], 'invalid patch_type parameter'
        self.patch_type = patch_type
        self.gradient = gradient
        self.step = step
        self.params = params

    def get_patch(self, xy):
        if self.patch_type in ['/', '//', '-', '--', '+', '++']:
            raise Exception('No patch to return for this patch type.')
        elif self.patch_type == 'circle':
            return Circle(
                xy, self.params['radius'],
                color='black'
            )
        elif self.patch_type == 'rectangle':
            return Rectangle(
                xy, self.params['width'],
                self.params['height'],
                self.params['angle'],
                color='black'
            )
        elif self.patch_type == 'ellipse':
            return Ellipse(
                xy, self.params['width'],
                self.params['height'],
                self.params['angle'],
                color='black'
            )
        elif self.patch_type == 'arc':
            return Arc(
                xy, self.params['width'],
                self.params['height'],
                self.params['angle'],
                color='black'
            )
        elif self.patch_type == 'pentagon':
            return CirclePolygon(
                xy, self.params['radius'],
                5,
                color='black'
            )
        elif self.patch_type == 'wedge':
            return Wedge(
                xy, self.params['radius'], 0,
                self.params['theta2'],
                color='black'
            )
        elif self.patch_type == 'arrow':
            x, y = xy
            return Arrow(
                x, y, self.params['dx'],
                self.params['dy'],
                self.params['width'],
                color='black'
            )

def get_base_image(height, width, color, shape, gradient=None):
    """

    :param height:
    :param width:
    :param color:
    :param shape:
    :param gradient:
    :return:
    """
    assert gradient in ['left', 'right', 'up', 'down', None]
    # initialize image and mask
    img = np.zeros(shape=(height, width, 3))
    mask = np.ones(shape=(height, width, 3))
    # find box window of the shape
    x_min = int(min([s[0] for s in shape]))
    x_max = int(max([s[0] for s in shape]))
    y_min = int(min([s[1] for s in shape]))
    y_max = int(max([s[1] for s in shape]))
    dx = x_max - x_min
    dy = y_max - y_min
    # Step through and determine mask at each pixel
    # in our shape window
    for i in range(dy):
        for j in range(dx):
            if gradient == 'right':
                mask[y_min + i, x_min + j] = j / dx
            elif gradient == 'left':
                mask[y_min + i, x_min + j] = 1 - j / dx
            elif gradient == 'up':
                mask[y_min + i, x_min + j] = 1 - i / dy
            elif gradient == 'down':
                mask[y_min + i, x_min + j] = i / dy

    # Now step through and apply the mask
    for i in range(height):
        for j in range(width):
            img[i, j] = mask[i, j] * color

    return img

def generate_texture(patch_type=None, gradient='sample', image_size=500):
    """

    :param patch_type:
    :param gradient:
    :param image_size:
    :return:
    """
    if patch_type is None:
        # randomly sample a patch type
        patch_types = [
            'ellipse', 'arc', 'arrow', 'circle',
            'rectangle', 'wedge', 'pentagon',
            '/', '//', '-', '--', '+', '++'
        ]
        patch_type = np.random.choice(patch_types)
    # Patch size will be uniformly sampled. Let's define
    # reasonable boundaries here
    patch_min_size = int(0.03*image_size)
    patch_max_size = int(0.09*image_size)
    # Now, build the texture object according to the specified type
    if patch_type == 'circle':
        params = {
            'radius': randint(patch_min_size, patch_max_size)
        }
    elif patch_type in ['rectangle', 'ellipse', 'arc']:
        params = {
            'height': randint(patch_min_size, patch_max_size),
            'width': randint(2*patch_min_size, 2*patch_max_size),
            'angle': randint(0, 180)
        }
    elif patch_type == 'pentagon':
        params = {
            'radius': randint(patch_min_size, patch_max_size)
        }
    elif patch_type == 'wedge':
        params = {
            'radius': randint(patch_min_size, patch_max_size),
            'theta2': randint(45, 270)
        }
    elif patch_type == 'arrow':
        params = {
            'dx': randint(-patch_max_size, patch_max_size),
            'dy': randint(-patch_max_size, patch_max_size),
            'width': randint(3*patch_min_size, 3*patch_max_size)
        }
    elif patch_type in ['/', '//', '-', '--', '+', '++']:
        params = None
    else:
        raise Exception('Invalid patch_type parameter.')

    if patch_type in ['ellipse', 'arc', 'arrow', 'circle','rectangle',
                      'wedge', 'pentagon']:
        # As another parameter, step size (space between
        # patches) will be uniformly sampled. Define
        # boundaries here
        step_min_size = int(0.15*image_size)
        step_max_size = int(0.30*image_size)
        # Now sample the step size. Later, we will step
        # through the image placing the patches at different
        # locations, 'step' pixels apart from one another
        step = randint(step_min_size, step_max_size)
    else:
        step = None

    if gradient == 'sample':
        # As a final parameter, we will sample a gradient from
        # a set of 5 options
        gradient_options = [None, 'left', 'right', 'up', 'down']
        gradient = np.random.choice(gradient_options)
    else:
        # In this case, the gradient has already been provided.
        # Let's error-check it.
        assert gradient in [None, 'left', 'right', 'up', 'down']

    # Now create the texture object instance and return
    return Texture(patch_type, gradient, step, params)

def add_texture(ax, texture, image_size=500):
    """

    :param ax:
    :param texture:
    :param image_size:
    :return:
    """
    if texture.patch_type in ['/', '//', '-', '--', '+', '++']:
        square = Polygon([(0, 0), (image_size, 0), (image_size, image_size),
                          (0, image_size)],
                         closed=True, fill=False, color=(0, 0, 0))
        square.set_hatch(texture.patch_type)
        ax.add_patch(square)
    else:
        for i in range(10, 2*image_size, texture.step):
            for j in range(10, 2*image_size, texture.step):
                patch = texture.get_patch((i,j))
                ax.add_patch(patch)