from __future__ import annotations

import os
from pathlib import Path
from black import out

import numpy as np
from matplotlib import pyplot as plt

from core import cat_aligner, config, utils

from PIL import ImageOps, Image

os.chdir(utils.Paths.ROOT_DIR / "notebooks")


def earliest_files(f: str) -> Path:
    def key(path: Path) -> int:
        return int(path.stem[: path.stem.find("_")])

    files = sorted(Path("images", f, "gif_subimages").iterdir(), key=key)
    print(f"_OUTPUTS {f}")
    for nf in files[:3]:
        print(f"## {nf.stem}")
        print(f"![my image]({nf})")


for f in (
    "simple_100",
    "simple",
    "eyes",
    "lstsq",
):
    earliest_files(f)
