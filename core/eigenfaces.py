from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from PIL import Image

from core import utils, cat_aligner, config


def dilate_components(arr: np.ndarray) -> np.ndarray:
    ret = arr.copy()
    ret = ret - ret.min()
    ret = ret / ret.max()
    return ret


def main():
    N_SAMPLES = 100
    ALIGNER: cat_aligner.CatAligner = cat_aligner.CatAlignerCropOnly
    out_folder = Path(os.path.expanduser(config.ROOT_DIR), "output", "gif_temp")
    out_folder.mkdir(exist_ok=True, parents=True)

    all_files = list(utils.Paths.gen_files())
    files = all_files[:N_SAMPLES]
    images, _ = zip(*[ALIGNER.transform(f) for f in files])
    names = [f.image.stem for f in files]
    shape = np.array(images[0]).shape

    X_train = np.array([np.array(im).flatten() for im in images])
    ret = utils.plot_portraits(X_train, shape, 4, 4, show=False)
    ret.savefig(out_folder / "sample_processed_images.png")

    X_mean = X_train.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

    fig = utils.plot_principle_components(S)
    fig.savefig(out_folder / "Principle Components")

    eigenfaces = np.array([dilate_components(arr) for arr in Vt])
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    ret = utils.plot_portraits(eigenfaces, shape, 4, 4, eigenface_titles)
    ret.savefig(out_folder / "eigenfaces.png")

    def reconstruct_with_components(image: int | slice, component: int | slice):
        principle_compoents = U[image, component] * S[component]
        principle_directions = Vt[component]

        return (principle_compoents @ principle_directions) + X_mean

    gif_folder = out_folder / "gif_subimages"
    gif_folder.mkdir()

    S_cum = S.cumsum() / S.sum()

    percentiles = [1]
    for x in np.arange(0, 1, 0.02):
        idx = np.searchsorted(S_cum, x)
        if idx > percentiles[-1]:
            percentiles.append(idx)

    for n_components in percentiles:
        X_final = reconstruct_with_components(slice(10, 30), slice(n_components))
        fig = utils.plot_portraits(
            X_final,
            shape,
            4,
            4,
            titles=[f"Cat {x}" for x in range(1, 17)],
            suptitle=f"{n_components} Components",
            show=False,
        )
        out_folder.mkdir(exist_ok=True)
        fig.savefig(gif_folder / f"{n_components}_components.png")


if __name__ == "__main__":
    main()
