#%%
import typing as tp
import math

import numpy as np
from matplotlib import pyplot as plt

from core import utils

files = utils.Paths.gen_files()
file = next(files)

image = utils.parse_image(file.image)

h, w, c = image.shape

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def imshow_array(arr: np.ndarray, points: tp.Iterable[utils.Point] = ()) -> None:
    fig, ax = plt.subplots(1)
    ax.imshow(arr)
    for point in points:
        circ = plt.Circle((point.x, point.y), 30)
        ax.add_patch(circ)


eyes = utils.Point.to_np(file.points[:2])
desired = np.array([[206, 306], [256, 256]])
x = np.linalg.solve(eyes, desired)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# https://en.wikipedia.org/wiki/Rotation_matrix
def rotate(points: tp.Iterable[utils.Point], theta: float):
    rotation_matrix = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), -math.cos()]
    )


"""
    eye1 eye2
x [[500, 245],
y  [603, 435]])

maybe we want 512x512 images, x1 y1

so we want maybe 
x [[206, 306],
y  [256, 256]])
"""


# %%
