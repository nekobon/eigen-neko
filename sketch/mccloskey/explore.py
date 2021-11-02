#%%
from __future__ import annotations
import os

ROOT_DIR = Path("~/src/eigen-neko").expanduser()
os.chdir(ROOT_DIR)

from pathlib import Path

from matplotlib import pyplot as plt

from core import utils


def show_image(path: Path):
    plt.imshow(utils.parse_image(path))


def get_output_path(path: Path, output_subdir=None, suffix=""):
    assert path.parts[-3] == "input"
    input_subdir = path.parts[-2]
    output_dir = utils.OUTPUT_PATH
    if output_subdir:
        output_dir = output_dir / output_subdir
    output_dir = output_dir / input_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{path.stem}{suffix}{path.suffix}"
    return out_path


## MAIN ##
files = utils.gen_files()
for i in range(100):
    file: utils.AnnotatedImage = next(files)
    face = file.extract_face(margin_x_min=10, margin_x_max=10, margin_y_max=20)
    outpath = get_output_path(file.image, output_subdir="extracted_faces_small_margin")
    plt.imsave(outpath, face)
print("done")

# %%

# %%
