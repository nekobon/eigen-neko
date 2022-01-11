#%%
import typing as tp
import math
import itertools as it
from typing_extensions import Annotated
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

from core import utils

files = list(utils.Paths.gen_files())


def gen_image_arrays(files: tp.Iterable[utils.AnnotatedImage], output_shape=(64, 64)):
    for file in files:
        image = Image.open(file.image)
        min_pt, max_pt = utils.Point.to_min_max(file.points)

        yield np.array(
            image.crop((min_pt.x, min_pt.y, max_pt.x, max_pt.y)).resize(output_shape)
        )


X_train = np.array(list(gen_image_vectors(files[:200])))
pca = decomposition.PCA()
pca.fit(X_train)


def expand_components(arr: np.ndarray):
    ret = arr
    ret -= ret.min()
    ret /= ret.max()
    return ret.reshape(shape)


# %%
