from __future__ import division, print_function
import numpy as np
import functools
from random import randint
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
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
    """

    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param x_offset:
    :param y_offset:
    :return:
    """
    # Sample a number of points for the polygon
    nb_points = randint(3, 10)
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
    #     # Now center the points by computing the mean distance
    #     # from the center and then subtracting this mean
    #     x_mean = np.mean([p[0] - (x_max-x_min)/2 for p in points])
    #     y_mean = np.mean([p[1] - (y_max-y_min)/2 for p in points])
    #     points = [(p[0]-x_mean, p[1]-y_mean) for p in points]

    return points


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

def generate_texture(patch_type=None, image_size=500):
    """

    :param patch_type:
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

    # As a final parameter, we will sample a gradient from
    gradient_options = [None, 'left', 'right', 'up', 'down']
    gradient = np.random.choice(gradient_options)

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

def generate_dataset_parameters(nb_categories, image_size=500):
    """

    :param nb_shapes:
    :return:
    """
    # Generate shapes, which are sets of points for which polygons will
    # be generated
    shapes = [generate_random_shape(0, 500, 0, 500, 100) for _ in
              range(nb_categories)]
    # Generate colors, which are 3-D vectors of values between 0-1 (RGB values)
    colors = generate_colors(nb_categories)
    # Generate textures
    patch_types = [
            'ellipse', 'arc', 'arrow', 'circle',
            'rectangle', 'wedge', 'pentagon'
        ]
    nb_variations = max(int(np.ceil(nb_categories / len(patch_types))), 3)
    textures = []
    for patch_type in patch_types:
        t_list = [generate_texture(patch_type, image_size)
                  for _ in range(nb_variations)]
        textures.extend(t_list)
    hatch_types = ['/', '//', '-', '--', '+', '++']
    for hatch_type in hatch_types:
        textures.append(Texture(hatch_type, gradient=None))
        textures.append(Texture(hatch_type, gradient='right'))
        textures.append(Texture(hatch_type, gradient='left'))
    textures = np.random.choice(textures, nb_categories, replace=False)

    return shapes, colors, textures

def generate_image(shape, color, texture, save_file):
    """

    :param shape:
    :param color:
    :param texture:
    :param save_file:
    :return:
    """
    # Generate the base image and save it to a file
    img = get_base_image(500, 500, color, shape, gradient=texture.gradient)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, interpolation='bicubic')
    add_texture(ax, texture, 500)
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