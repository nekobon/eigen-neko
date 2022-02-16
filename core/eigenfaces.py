from __future__ import annotations

import os
from pathlib import Path
from black import out

import numpy as np
from matplotlib import pyplot as plt

from core import cat_aligner, config, utils

from PIL import ImageOps, Image


def dilate_components(arr: np.ndarray) -> np.ndarray:
    ret = arr.copy()
    ret = ret - ret.min()
    ret = ret / ret.max()
    return ret


# NOTE: Images are not getting made appropriately. Also
# /home/mccloskey/src/eigen-neko/core/utils.py:138: RuntimeWarning: More than 20 figures have been opened.
# Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly
# closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
# fig = plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
def save_gif(*, outpath: Path, subimage_dir: Path) -> Path:
    def key(path: Path) -> int:
        return int(path.stem[: path.stem.find("_")])

    files = sorted(subimage_dir.iterdir(), key=key)
    img, *imgs = [Image.open(f) for f in files]

    img.save(
        fp=outpath,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=200,
        loop=0,
    )

    return outpath


def main(*, aligner: cat_aligner.CatAligner, n_samples: int, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"Outputting to {out_folder}")

    all_files = utils.Paths.list_sorted_files()
    files = all_files[:n_samples]
    names = [f.image.stem[4:] for f in files]
    images = [ImageOps.grayscale(aligner.align_one_image(f, 64, 64)) for f in files]
    shape = np.array(images[0]).shape

    X_train = np.array([np.array(im).flatten() for im in images])
    ret = utils.plot_portraits(X_train, shape, 4, 8, titles=names)
    ret.savefig(out_folder / "sample_processed_images.png")
    plt.close(ret)

    X_mean = X_train.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)

    fig = utils.plot_principle_components(S)
    fig.savefig(out_folder / "Principle Components")
    plt.close(fig)

    eigenfaces = np.array([dilate_components(arr) for arr in Vt])
    eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
    ret = utils.plot_portraits(eigenfaces, shape, 4, 8, eigenface_titles)
    ret.savefig(out_folder / "eigenfaces.png")
    plt.close(ret)

    def reconstruct_with_components(
        image: int | slice | list[int], component: int | slice
    ):
        principle_compoents = U[image, component] * S[component]
        principle_directions = Vt[component]

        return (principle_compoents @ principle_directions) + X_mean

    gif_folder = out_folder / "gif_subimages"
    gif_folder.mkdir(exist_ok=True)

    S_cum = S.cumsum() / S.sum()

    percentiles = [1]
    for x in np.arange(0, 1, 0.02):
        idx = np.searchsorted(S_cum, x)
        if idx > percentiles[-1]:
            percentiles.append(idx)

    for n_components in percentiles:

        X_final = reconstruct_with_components(slice(32), slice(n_components))
        fig = utils.plot_portraits(
            X_final, shape, 4, 8, titles=names, suptitle=f"{n_components} Components"
        )
        out_folder.mkdir(exist_ok=True)
        fig.savefig(gif_folder / f"{n_components}_components.png")

    save_gif(outpath=out_folder / "iterative_components.gif", subimage_dir=gif_folder)


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif


if __name__ == "__main__":
    main()
