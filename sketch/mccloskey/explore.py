#%%
from __future__ import annotations
from pathlib import Path
import os
import typing as tp
from matplotlib import pyplot as plt
from matplotlib import image as mpl_image

root_dir = Path("~/src/eigen-neko").expanduser()
os.chdir(root_dir)
input_path = root_dir / "input"
output_path = root_dir / "output"

class InputPair(tp.NamedTuple):
    image: Path
    annotation: Path

class Point(tp.NamedTuple):
    x: int
    y: int

    @staticmethod
    def to_min_max(points: tp.Iterable[Point]):
        xs, ys = zip(*points)
        return Point(min(xs), min(ys)), Point(max(xs), max(ys))

    @classmethod
    def box_slice(cls, p1, p2, margin_x_min=0, margin_y_min=0, margin_x_max=0, margin_y_max=0):
        min_x, max_x = sorted([p1.x, p2.x])
        min_y, max_y = sorted([p1.y, p2.y])

        return (
            slice(max(0, min_x - margin_x_min), max_x + margin_x_max), 
            slice(max(0, min_y - margin_y_min), max_y + margin_y_max)
        )

class AnnotatedImage(tp.NamedTuple):
    image: Path
    points: tp.Tuple[Point, ...]

    @classmethod
    def from_paths(cls, annotation: Path, image: Path):
        num_points, *points = (int(x) for x in annotation.read_text().strip().split())
        assert num_points*2 == len(points)
        points = [Point(points[x-1], points[x]) for x in range(1, len(points), 2)]
        return cls(
            image=image,
            points=points,
        )

    def extract_face(self, margin_x_min=0, margin_y_min=0, margin_x_max=0, margin_y_max=0):
        vector = parse_image(self.image)
        min_pt, max_pt = Point.to_min_max(self.points)
        slice_x, slice_y = Point.box_slice(min_pt, max_pt, margin_x_min=margin_x_min, margin_y_min=margin_y_min, margin_x_max=margin_x_max, margin_y_max=margin_y_max)
        # vector has inverted x y
        return vector[slice_y, slice_x]
        







def gen_files():
    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            path = Path(dirname, filename)
            if path.suffix == '.jpg':
                yield AnnotatedImage.from_paths(image=path, annotation=path.with_suffix('.jpg.cat'))

def parse_image(path: Path):
    # top -> bottom - x
    # left -> right - y
    return mpl_image.imread(path)

def show_image(path: Path):
    plt.imshow(parse_image(path))

def get_output_path(path: Path, output_subdir=None, suffix=''):
    assert path.parts[-3] == 'input'
    input_subdir = path.parts[-2]
    output_dir = output_path
    if output_subdir:
        output_dir = output_dir / output_subdir
    output_dir = output_dir / input_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = (output_dir / f'{path.stem}{suffix}{path.suffix}')
    return out_path

## MAIN ##
files = gen_files()
for i in range(100):
    file = next(files)
    face = file.extract_face(margin_x_min=10, margin_x_max=10, margin_y_max=20)
    outpath = get_output_path(file.image, output_subdir='extracted_faces_small_margin')
    plt.imsave(outpath, face)



# %%

# %%
