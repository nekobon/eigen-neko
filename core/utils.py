from __future__ import annotations

import os
import typing as tp
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import figure
from matplotlib import image as mpl_image
from matplotlib import pyplot as plt

from core import config


class Paths:
    ROOT_DIR = Path(config.ROOT_DIR).expanduser()
    INPUT_PATH = ROOT_DIR / "input"
    OUTPUT_PATH = ROOT_DIR / "output"

    @classmethod
    def gen_files(
        cls,
        input_path: PathSpecifier = INPUT_PATH,
    ) -> tp.Iterator[AnnotatedImage]:
        """
        Args:
            Dataset directory

        Returns:
            Iterator of AnnotatedImages (image_path, points) pairs
        """
        for dirname, _, filenames in os.walk(input_path):
            for filename in filenames:
                path = Path(dirname, filename)
                if path.suffix == ".jpg":
                    yield AnnotatedImage.from_paths(
                        image=path, annotation=path.with_suffix(".jpg.cat")
                    )


PathSpecifier = tp.Union[str, Path]


class Point(tp.NamedTuple):
    x: int
    y: int

    @staticmethod
    def to_min_max(points: tp.Iterable[Point]):
        xs, ys = zip(*points)
        return Point(min(xs), min(ys)), Point(max(xs), max(ys))

    @classmethod
    def box_slice(
        cls, p1, p2, margin_x_min=0, margin_y_min=0, margin_x_max=0, margin_y_max=0
    ):
        min_x, max_x = sorted([p1.x, p2.x])
        min_y, max_y = sorted([p1.y, p2.y])

        return (
            slice(max(0, min_x - margin_x_min), max_x + margin_x_max),
            slice(max(0, min_y - margin_y_min), max_y + margin_y_max),
        )

    @staticmethod
    def to_np(points: tp.Sequence[Point]) -> np.ndarray:
        # first column x, second column y
        return np.array(points)

    def __sub__(self, other: tp.Any) -> Point:
        if not isinstance(other, Point):
            return NotImplemented

        return Point(x=self.x - other.x, y=self.y - other.y)


class AnnotatedImage(tp.NamedTuple):
    image: Path
    points: tp.Tuple[Point, ...]

    @classmethod
    def from_image_path(cls, path: Path):
        return cls.from_paths(image=path, annotation=path.with_suffix(".jpg.cat"))

    @classmethod
    def from_paths(cls, annotation: Path, image: Path):
        num_points, *points = (int(x) for x in annotation.read_text().strip().split())
        assert num_points * 2 == len(points)
        points = [Point(points[x - 1], points[x]) for x in range(1, len(points), 2)]
        return cls(
            image=image,
            points=points,
        )

    # similar functionality in cat_aligner.CatAligner.truncate_by_all_points
    # def extract_face(
    #     self, margin_x_min=0, margin_y_min=0, margin_x_max=0, margin_y_max=0
    # ):
    #     vector = parse_image(self.image)
    #     min_pt, max_pt = Point.to_min_max(self.points)
    #     slice_x, slice_y = Point.box_slice(
    #         min_pt,
    #         max_pt,
    #         margin_x_min=margin_x_min,
    #         margin_y_min=margin_y_min,
    #         margin_x_max=margin_x_max,
    #         margin_y_max=margin_y_max,
    #     )
    #     # vector has inverted x y
    #     return vector[slice_y, slice_x]


def parse_image(path: Path):
    # top -> bottom - x
    # left -> right - y
    return mpl_image.imread(path)


def plot_image(
    image_arr,
    shape,
    axis: plt.Axes = None,
    title="",
):
    axis = plt if axis is None else axis

    axis.imshow(image_arr.reshape(shape), cmap="gray")
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_portraits(
    images, shape, n_row, n_col, titles=None, suptitle=None, show=True
) -> figure.Figure:
    titles = [""] * n_row * n_col if titles is None else titles
    fig = plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.20)
    for i in range(n_row * n_col):
        axis = plt.subplot(n_row, n_col, i + 1)
        plot_image(images[i], shape, axis, titles[i])
    if suptitle:
        plt.suptitle(suptitle)
    if show:
        plt.show()

    return fig


def plot_principle_components(S: np.ndarray) -> figure.Figure:
    sns.scatterplot(x=range(1, len(S) + 1), y=S.cumsum() / S.sum())
    S_cum = S.cumsum() / S.sum()
    vline_y = [x / 10 for x in range(1, 10)]
    vline_x = [np.searchsorted(S_cum, x) for x in vline_y]
    plt.ylim(0, 1)
    plt.xlim(0, len(S))
    plt.xlabel("Num Components")
    plt.ylabel("Cumulative Sum")
    plt.title("Component Descriptive Power")
    plt.vlines(
        x=vline_x,
        ymin=0,
        ymax=vline_y,
        colors="teal",
        ls="--",
        lw=2,
        label="Number of Components for each decile",
    )
    plt.hlines(y=vline_y, xmin=0, xmax=vline_x, colors="teal", ls="--", lw=2)
    plt.legend(bbox_to_anchor=(1, -0.15))

    return plt.gcf()
