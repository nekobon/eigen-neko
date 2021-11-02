#%%
import typing as tp

import numpy as np
from matplotlib import pyplot as plt

from core import utils

files = utils.Paths.gen_files()
file = next(files)

image = utils.parse_image(file.image)

h, w, c = image.shape


def imshow_array(arr: np.ndarray, points: tp.Iterable[utils.Point] = ()) -> None:
    fig, ax = plt.subplots(1)
    ax.imshow(arr)
    for point in points:
        circ = plt.Circle((point.x, point.y), 50)
        ax.add_patch(circ)


eyes = utils.Point.to_np(file.points[:2])


# %%
