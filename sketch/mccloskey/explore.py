from pathlib import Path
import os
import typing as tp
import matplotlib
import json

root_dir = Path("~/src/eigen-neko").expanduser()
os.chdir(root_dir)
input_path = root_dir / "input"

class InputPair(tp.NamedTuple):
    image: Path
    annotation: Path

class Point(tp.NamedTuple):
    x: int
    y: int
class AnnotatedImage(tp.NamedTuple):
    image: Path
    points: tp.Tuple[Point, ...]

    @classmethod
    def from_input_pair(cls, input_pair: InputPair):
        num_points, *points = (int(x) for x in input_pair.annotation.read_text().strip().split())
        assert num_points*2 == len(points)
        points = [Point(points[x-1], points[x]) for x in range(1, len(points)-1)]
        return cls(
            image=input_pair.image,
            points=points,
        )

def gen_files():
    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            path = Path(dirname, filename)
            if path.suffix == '.jpg':
                yield InputPair(path, path.with_suffix('.jpg.cat'))

def parse_image(path: Path):
    return matplotlib.image.imread(path)

def show_image(path: Path):
    matplotlib.pyplot.imshow(parse_image(path))

if __name__ == "__main__":
    files = gen_files()
    file = next(files)
    img = AnnotatedImage.from_input_pair(file)



    from IPython import embed
    embed()
