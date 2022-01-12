# From https://towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184
# %%
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import typing as tp
from core import utils
import typing as tp
import math
import itertools as it
from typing_extensions import Annotated
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt

from core import utils

# %%


class PCAResult(tp.NamedTuple):
    mean: np.ndarray
    centered_data: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray


N_SAMPLES = 500
# """It helps visualising the portraits from the dataset."""
def plot_portraits(images, titles, shape, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(shape), cmap="gray")
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()


def gen_images(
    files: tp.Iterable[utils.AnnotatedImage],
    output_shape=(64, 64),
    gray=True,
    crop=True,
) -> tp.Iterable[Image.Image]:
    for file in files:
        image = Image.open(file.image)

        if gray:
            image = ImageOps.grayscale(image)

        if crop:
            min_pt, max_pt = utils.Point.to_min_max(file.points)
            image = image.crop((min_pt.x, min_pt.y, max_pt.x, max_pt.y))

        if output_shape:
            image = image.resize(output_shape)

        yield image


def pca(X) -> PCAResult:
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

    return PCAResult(mean=mean, centered_data=centered_data, U=U, S=S, Vt=Vt)


def pca_from_components(pca_result: PCAResult, key_or_slice: int | slice):
    principle_compoents = pca_result.U[:, key_or_slice] * pca_result.S[key_or_slice]
    principle_directions = pca_result.Vt[key_or_slice]

    return (principle_compoents @ principle_directions) + pca_result.mean


def dilate_components(arr: np.ndarray) -> np.ndarray:
    ret = arr.copy()
    ret -= ret.min()
    ret /= ret.max()
    return ret


def reconstruction(pca_result: PCAResult, shape, image_index, n_pc):
    components = pca_result.Vt[:n_pc]
    n_samples, n_features = pca_result.centered_data.shape
    weights = np.dot(pca_result.centered_data, components)
    centered_vector = np.dot(weights[image_index, :], components)
    recovered_image = (pca_result.mean + centered_vector).reshape(shape)
    return recovered_image


# %% Gather Images

files = list(utils.Paths.gen_files())
images = list(gen_images(files[:N_SAMPLES]))
names = [f.image.stem for f in files[:N_SAMPLES]]
shape = np.array(images[0]).shape

X_train = np.array([np.array(im).flatten() for im in images])

plot_portraits(X_train, names, shape, n_row=4, n_col=4)


# %% Run PCA

n_components = 100
pca_result = pca(X_train)
# columns of Vt (rows of V) are the principle directions
eigenfaces = np.array([dilate_components(arr) for arr in pca_result.Vt])
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_portraits(eigenfaces, eigenface_titles, shape, 4, 4)

# %%
# Add progressive compoents
gradual_pca = [
    pca_from_components(pca_result, slice(0, i + 1, None)) for i in range(16)
]
gradual_pca_i = lambda i: [arr[i] for arr in gradual_pca]
plot_portraits(gradual_pca[2], names, shape, n_row=4, n_col=4)


# %% Construct


recovered_images = [
    reconstruction(pca_result, shape, i, n_components) for i in range(N_SAMPLES)
]
plot_portraits(recovered_images, names, shape, n_row=4, n_col=4)

# %%
