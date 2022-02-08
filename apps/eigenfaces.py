from argparse import ArgumentParser
from ctypes import alignment
from core import eigenfaces, cat_aligner, utils

ALIGNERS = (
    cat_aligner.CatAlignerEyes,
    cat_aligner.CatAlignerLSTSQ,
    cat_aligner.CatAlignerSimple,
)

NAME_TO_ALIGNER = {aligner.__name__: aligner for aligner in ALIGNERS}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Daily performance report.")
    parser.add_argument(
        "--aligner",
        choices=NAME_TO_ALIGNER.keys(),
        required=True,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        help=f"Subdirectory of {utils.Paths.OUTPUT_PATH}",
    )

    return parser


def main() -> None:
    options = get_parser().parse_args()
    eigenfaces.main(
        aligner=NAME_TO_ALIGNER[options.aligner],
        n_samples=options.n_samples,
        out_folder=utils.Paths.OUTPUT_PATH / options.out_folder,
    )


if __name__ == "__main__":
    main()
