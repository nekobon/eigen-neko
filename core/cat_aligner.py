from core.utils import Path, AnnotatedImage
import typing as tp
import itertools
from core import utils
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from PIL import Image


def show_image(path: Path):
    plt.imshow(utils.parse_image(path))


def _show_cat_and_points(cat: AnnotatedImage) -> None:

    show_image(cat.image)

    cat_img = utils.parse_image(cat.image)

    for point in cat.points:
        # NOTE that x and y are flipped!
        cat_img[point.y - 10 : point.y + 10, point.x - 10 : point.x + 10] = [
            0,
            0,
            255,
        ]

    plt.imshow(cat_img)


def stack_images(images: tp.Sequence["Image"]):
    """assumes all same size"""
    w, h = images[0].size
    rt = np.sqrt(len(images))
    n_col = int(rt)
    n_row = int(n_col + int((rt - n_col) > 0))

    new_img = Image.new("RGB", (n_col * w, n_row * h))

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        new_img.paste(img, (x_offset, y_offset))

        if (i + 1) % n_col:
            x_offset += w
        else:
            x_offset = 0
            y_offset += h
        # print(x_offset, y_offset)

    return new_img


def truncate(
    img: Image,
    eye_points: tp.List[tp.List[int]],
    width: int,
    height: int,
) -> Image:
    eye_w_to_half_height = 1.3 * height / width
    eye_w_to_half_width = 1.3
    eye_height = 0.8  # 0 = at bototm, 1 = at middle, 2 = at top

    left, right = eye_points
    mid_eye_x = (left[0] + right[0]) / 2
    mid_eye_y = (left[1] + right[1]) / 2
    eye_w = right[0] - left[0]
    x_from_center = eye_w * eye_w_to_half_width
    y_from_center = eye_w * eye_w_to_half_height

    left = mid_eye_x - x_from_center
    right = mid_eye_x + x_from_center
    top = mid_eye_y - (y_from_center * (2 - eye_height))
    bottom = mid_eye_y + (y_from_center * eye_height)

    cropped = img.crop((left, top, right, bottom))
    # cropped.show()
    # return cropped
    scaled = cropped.resize((width, height))
    # scaled.show()
    return scaled


class CatAlignerLSTSQ:
    @staticmethod
    def get_standard_cat() -> AnnotatedImage:
        standard_cat_path = utils.Paths.INPUT_PATH / "CAT_00" / "00000055_003.jpg"
        return AnnotatedImage.from_image_path(standard_cat_path)

    @staticmethod
    def get_transforming_matrix(standard_cat: AnnotatedImage, test_cat: AnnotatedImage):
        standard_points = utils.Point.to_np(standard_cat.points)[
            :, [1, 0]
        ]  # flip x and y

        test_points = utils.Point.to_np(test_cat.points)[:, [1, 0]]  # flip x and y

        col_with_ones = np.ones((9, 1))

        # add column of 1s to make all transformation done by multiplication
        # this supports translation
        standard = np.append(standard_points, col_with_ones, 1)
        test = np.append(test_points, col_with_ones, 1)

        x, residuals, rank, singlar = linalg.lstsq(
            a=test, b=standard, rcond=None
        )  # a @ x = b
        return x

    @staticmethod
    def _get_coordinates(test_cat_image: np.array):
        h, w, _ = test_cat_image.shape
        points_gen = itertools.product(range(1, h + 1), range(1, w + 1))

        coordinates_2d = np.array(list(points_gen))

        # add column with ones for translation
        return np.append(coordinates_2d, np.ones((h * w, 1)), 1).astype(int)

    @staticmethod
    def _get_image_multiplier(src_w: int, src_h: int, new_w: int, new_h: int) -> int:
        """we want to make the image smaller to avoid gaps in pixels"""
        w_ratio = src_w / new_w
        h_ratio = src_h / new_h
        print(f"{src_w=}")
        print(f"{new_w=}")
        print(f"{src_h=}")
        print(f"{new_h=}")
        print(f"{min(1, w_ratio, h_ratio)=}")

        return min(1, w_ratio, h_ratio) * 0.5

    @staticmethod
    def get_new_eye_points(cat: AnnotatedImage, x: np.array, size_multiplier: float):
        eyes = utils.Point.to_np(cat.points[:2])[:, [1, 0]]  # flip x and y
        eyes = np.append(eyes, np.ones((2, 1)), 1)
        return np.rint(eyes @ x * size_multiplier).astype(int)

    @classmethod
    def transform(
        cls, test_cat: AnnotatedImage
    ) -> tp.Tuple["Image", tp.List[tp.List[int]]]:

        standard_cat = cls.get_standard_cat()

        x = cls.get_transforming_matrix(standard_cat=standard_cat, test_cat=test_cat)

        test_cat_image = utils.parse_image(test_cat.image)
        coordinates_3d = cls._get_coordinates(test_cat_image)

        # subtract 1 to start from index 0
        destination_3d = np.rint(coordinates_3d @ x).astype(int)
        destination_3d = np.clip(
            destination_3d, a_min=0, a_max=None
        )  # remove outside boundary

        src_xy = coordinates_3d[:, :2] - 1
        new_xy = destination_3d[:, :2] - 1

        # separate into two vectors
        src_x, src_y = src_xy[:, 0], src_xy[:, 1]
        new_x, new_y = new_xy[:, 0], new_xy[:, 1]

        # get heights and widths
        src_w, src_h = src_x.max(), src_y.max()
        new_w, new_h = new_x.max(), new_y.max()

        size_multiplier = cls._get_image_multiplier(src_w, src_h, new_w, new_h)

        # scaled destination
        dst_x = (new_x * size_multiplier).astype(int)
        dst_y = (new_y * size_multiplier).astype(int)

        eye_points = cls.get_new_eye_points(test_cat, x, size_multiplier)
        final_image = np.zeros([dst_x.max() + 1, dst_y.max() + 1, 3]).astype(int) + 255
        final_image[dst_x, dst_y] = test_cat_image[src_x, src_y]

        img = Image.fromarray(final_image.astype(np.uint8))

        return img, eye_points[:, [1, 0]].tolist()


class CatAlignerEyes:
    @staticmethod
    def get_eye_rot_angle_and_size(test_cat: AnnotatedImage):
        # only using eyes (1st and 2nd points)
        p = test_cat.points  # list of (x, y) points
        x = p[1][0] - p[0][0]  # y distance between two eyes
        y = p[1][1] - p[0][1]  # x distance between two eyes

        # note that y counts from top to bottom
        return np.arctan(y / x), np.sqrt(x ** 2 + y ** 2)

    @classmethod
    def transform(
        cls, test_cat: AnnotatedImage
    ) -> tp.Tuple["Image", tp.List[tp.List[int]]]:

        img = Image.open(test_cat.image)
        angle_rad, eye_w = cls.get_eye_rot_angle_and_size(test_cat)
        angle_deg = angle_rad * 180 / np.pi

        left_eye_x, left_eye_y = test_cat.points[0][0], test_cat.points[0][1]
        rotated = img.rotate(angle_deg, center=(left_eye_x, left_eye_y))

        new_eye_points = [
            [left_eye_x, left_eye_y],
            [left_eye_x + eye_w, left_eye_y],
        ]
        # scale_mult = eye_w / eye_w_target
        # new_size = (
        #     int(rotated.size[0] * scale_mult),
        #     int(rotated.size[1] * scale_mult),
        # )
        # scaled = rotated.resize(new_size)

        return rotated, new_eye_points


def gen_aligned(n: int, aligner: type):
    gen = utils.Paths.gen_files()
    for i, cat in enumerate(gen):
        print(aligner, i)
        if i == n:
            return
        new_image, eyes = aligner.transform(cat)
        yield truncate(new_image, eyes, width=100, height=100)

    # plt.imshow(new_image)
    # img, points = Image.fromarray(new_image.astype(np.uint8))
    # show_image(cat.image)
    # plt.show()
    # continue
    # img.show()


n = 64

aligned_eyes = list(gen_aligned(n, aligner=CatAlignerEyes))
img_eyes = stack_images(aligned_eyes)
img_eyes.show()

aligned_lstsq = list(gen_aligned(n, aligner=CatAlignerLSTSQ))
img_lstsq = stack_images(aligned_lstsq)
img_lstsq.show()
