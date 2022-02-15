from __future__ import annotations

import json
from pathlib import Path
import typing as tp
from copy import deepcopy
from enum import Enum
from itertools import count


class Cell(tp.TypedDict, total=False):
    cell_type: str
    execution_count: int | None
    id: str
    metadata: dict[str, tp.Any]
    outputs: list[str]
    source: list[str]


class CellType(str, Enum):
    MARKDOWN = "markdown"
    CODE = "code"


class SlideType(str, Enum):
    SLIDE = "slide"
    SUBSLIDE = "subslide"
    FRAGMENT = "fragment"


class Notebook:
    BASE_NOTEBOOK = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "overall-anatomy",
                "metadata": {},
                "source": ["my markdown 1"],
            },
            {
                "cell_type": "markdown",
                "id": "russian-confusion",
                "metadata": {},
                "source": ["my markdown 2"],
            },
        ],
        "metadata": {
            "celltoolbar": "Slideshow",
            "kernelspec": {
                "display_name": "Python 3.8.12 64-bit ('.env-eigeneko': venv)",
                "language": "python",
                "name": "python3812jvsc74a57bd04eee3b84c16cd653c9409d307a1e0cdaeaa31ed4117853a4ee7c842a85681cd7",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    def __init__(self) -> None:
        self._dict = deepcopy(self.BASE_NOTEBOOK)
        self._id_gen = (str(x) for x in count())

    @property
    def cells(self) -> list[Cell]:
        return self._dict["cells"]

    def add_cell_markdown(
        self, *, source: list[str], slide_type: SlideType = SlideType.FRAGMENT
    ):
        cell = Cell(
            cell_type=CellType.MARKDOWN.value,
            id=next(self._id_gen),
            metadata={"slideshow": {"slide_type": slide_type.value}},
            source=source,
        )
        self.cells.append(cell)

    def add_cell_code(
        self, *, source: list[str], slide_type: SlideType = SlideType.FRAGMENT
    ):
        cell = Cell(
            cell_type=CellType.CODE.value,
            execution_count=None,
            id=next(self._id_gen),
            metadata={"slideshow": {"slide_type": slide_type.value}},
            outputs=[],
            source=source,
        )

        self.cells.append(cell)

    def save(self, fp: str) -> str:
        with open(fp, "w") as f:
            json.dump(self._dict, f, indent=2)
        return fp

    def add_lines(self, lines: tp.List[str]) -> None:
        code = False
        code_block: tp.List[str] = []
        for line in lines:
            line = line.strip("\n")
            flag = line[: line.find(" ")] if " " in line else line

            if flag == "CODE":
                code = True
                continue

            if flag == "END":
                self.add_cell_code(source=code_block)
                code_block = []
                code = False
                continue

            if code:
                code_block.append(f"{line}\n")
                continue

            if flag == "#":
                slide_type = SlideType.SLIDE
            elif flag == "##":
                slide_type = SlideType.SUBSLIDE
            else:
                slide_type = SlideType.FRAGMENT

            self.add_cell_markdown(source=line, slide_type=slide_type)

    def add_markdown_cells(self, fp) -> None:
        with open(fp) as f:
            lines = f.readlines()

        self.add_lines(lines)
        return nb

    def add_jupyter_cells(self, fp) -> None:
        with open(fp) as f:
            nb_json = json.load(f)

        self.cells.extend(nb_json["cells"])
        return nb


if __name__ == "__main__":
    outpath = Path("test_nb2.ipynb")
    nb = Notebook()
    nb.add_markdown_cells("sketch/mccloskey/presentation.md")
    nb.add_jupyter_cells("test_nb.ipynb")
    print(nb.save(outpath))
    print(
        f"jupyter nbconvert {outpath} --to slides; firefox {outpath.with_suffix('.slides.html')}"
    )
